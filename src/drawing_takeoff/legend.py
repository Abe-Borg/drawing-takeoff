"""Legend / recognition — the M3 vision step (the project's first LLM call).

Turns the geometry's anonymous line STYLES (pen color + width + dashes) into
named building systems, so a length total can be attributed ("Fire-protection
sprinkler pipe: 1,341 LF") and background/reference linework is excluded.

Design, per the user's guidance: **construction legends are unreliable** —
incomplete, on a different (lead) sheet, or contradicted by the drawing — so the
legend is treated as *advisory*. The model is given each style's geometry stats
(run count, measured length) plus optional rendered swatches and an optional
legend image, and asked to reconcile them; it flags styles it can't place rather
than guessing. Every :class:`SystemLabel` carries a ``confidence`` and an
``ambiguous`` flag so the engine totals only what it can stand behind.

Structured output is obtained via the API's enforced JSON-schema response
format (``output_config.format``), parsed from the reply's text block. A forced
tool call would also guarantee the schema, but forced ``tool_choice`` is
incompatible with thinking — and adaptive thinking (with the central effort
policy, ``api_config.PHASE_LABELING``) is what these vision+reasoning
judgments benefit from most. No PyMuPDF here — images arrive as PNG bytes and
legend pages as single-page PDF bytes (both produced by ``geometry``), and the
Anthropic client is duck-typed and injectable for hermetic tests.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, replace

from . import measure
from .core import api_config
from .models import SheetGeometry, StyleKey, SystemLabel

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a quantity-takeoff assistant for construction drawings. You are given \
the distinct line STYLES (pen color + width + dash pattern) used on one sheet, \
with geometry stats for each style (how many continuous runs, total measured \
length), optionally a rendered swatch per style, and optionally a legend/symbols \
image from the project's lead sheet.

For each style, decide:
  - `system`: the building system it represents (e.g. "Fire-protection sprinkler \
pipe", "Domestic cold water", "Architectural background").
  - `measurable`: true only if it is a LINEAR RUN a length takeoff should total \
(pipe / duct / conduit / wall). Background, dimension lines, leaders, text, \
hatching, and title-block linework are NOT measurable.

Construction legends are frequently incomplete, live on a different sheet, or are \
contradicted by the drawing itself — treat any legend as ADVISORY, not ground \
truth. Reconcile it against the geometry stats and the discipline: a style with \
many long runs in the sheet's primary system is probably that system's linework \
even if a legend omits it. When you cannot confidently place a style, set \
`ambiguous` true and `confidence` low rather than guessing — a human reviews \
flagged styles.

Reply with the labels JSON (the response format is enforced): a `labels` array \
with exactly one entry for every style id provided."""

_LABELS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "labels": {
            "type": "array",
            "description": "One entry per style id given in the prompt.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "style_id": {"type": "string"},
                    "system": {
                        "type": "string",
                        "description": "System name; use 'Background / non-takeoff' for reference linework.",
                    },
                    "measurable": {
                        "type": "boolean",
                        "description": "True only for a linear run to be totaled (pipe/duct/conduit/wall).",
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "ambiguous": {
                        "type": "boolean",
                        "description": "True if you could not confidently place it; flagged for human review.",
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["style_id", "system", "measurable", "confidence", "ambiguous", "reasoning"],
            },
        }
    },
    "required": ["labels"],
}


@dataclass
class _Candidate:
    style_id: str
    style_key: StyleKey
    n_runs: int
    total_ft: float | None
    total_pt: float
    longest_ft: float | None
    extent: tuple[float, float, float, float]


def _candidates(geometry: SheetGeometry, *, ppf: float | None, max_styles: int) -> list[_Candidate]:
    """The most takeoff-relevant styles, ranked by total measured length.

    Border/matchline runs are excluded so the sheet border never poses as a
    candidate "main" and the per-style lengths shown to the model are honest.
    """
    runs_by_style = measure.runs_by_style(geometry, ppf=ppf, exclude_border=True)
    rows: list[_Candidate] = []
    for style, runs in runs_by_style.items():
        if not runs:
            continue
        total_pt = sum(r.length_pt for r in runs)
        longest_pt = max(r.length_pt for r in runs)
        xs0 = min(r.bbox[0] for r in runs)
        ys0 = min(r.bbox[1] for r in runs)
        xs1 = max(r.bbox[2] for r in runs)
        ys1 = max(r.bbox[3] for r in runs)
        rows.append(
            _Candidate(
                style_id="",
                style_key=style,
                n_runs=len(runs),
                total_ft=(total_pt / ppf) if ppf else None,
                total_pt=total_pt,
                longest_ft=(longest_pt / ppf) if ppf else None,
                extent=(xs0, ys0, xs1, ys1),
            )
        )
    rows.sort(key=lambda c: -c.total_pt)
    rows = rows[:max_styles]
    for i, c in enumerate(rows):
        c.style_id = f"s{i}"
    return rows


def _style_line(c: _Candidate) -> str:
    k = c.style_key
    color = "none" if k.stroke_color is None else ",".join(f"{v:.2f}" for v in k.stroke_color)
    length = f"{c.total_ft:,.0f} ft" if c.total_ft is not None else f"{c.total_pt:,.0f} pt"
    longest = f"{c.longest_ft:.1f} ft" if c.longest_ft is not None else "-"
    w = c.extent[2] - c.extent[0]
    h = c.extent[3] - c.extent[1]
    return (
        f"  {c.style_id}: stroke=[{color}] width={k.width} dashes={k.dashes!r}; "
        f"{c.n_runs} runs, {length} total, longest run {longest}, "
        f"spread {w:.0f}x{h:.0f} pt"
    )


def _image_block(png_bytes: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_bytes).decode("ascii"),
        },
    }


def _document_block(pdf_bytes: bytes) -> dict:
    """Native-PDF content block: the model gets the page image AND its text layer."""
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.standard_b64encode(pdf_bytes).decode("ascii"),
        },
    }


_FILES_API_BETA = "files-api-2025-04-14"


def _request_kwargs(*, model: str, system: str, content: list[dict], schema: dict, client=None) -> dict:
    """Assemble the structured-output labeling request (shared by M3 and M7).

    The JSON-schema response format rides ``output_config`` next to the
    central effort policy, and adaptive thinking is attached per
    ``api_config.PHASE_LABELING`` — the schema stays enforced without a forced
    tool call, which thinking does not allow. An unknown model id is resolved
    against the Models API first (once per process) so an env-pinned newer
    model gets its real capabilities. Content referencing a Files API
    ``file_id`` makes the request carry the files beta header.
    """
    api_config.ensure_model_registered(model, client=client)
    kwargs: dict = {
        "model": model,
        "max_tokens": api_config.phase_output_cap(api_config.PHASE_LABELING, model=model),
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }
    output_config = api_config.effort_config_for(model=model, phase=api_config.PHASE_LABELING) or {}
    output_config["format"] = {"type": "json_schema", "schema": schema}
    kwargs["output_config"] = output_config
    if any(
        isinstance(b, dict) and (b.get("source") or {}).get("type") == "file" for b in content
    ):
        kwargs["extra_headers"] = {"anthropic-beta": _FILES_API_BETA}
    return api_config.apply_thinking_config(kwargs, model=model, phase=api_config.PHASE_LABELING)


def upload_legend(client, *, legend_pdf: bytes | None = None, legend_image: bytes | None = None) -> dict | None:
    """Upload the advisory legend once via the Files API; return a reusable block.

    On a multi-sheet set the same legend bytes otherwise re-upload inside every
    per-sheet request; a one-time ``client.beta.files.upload`` turns that into
    a small ``file_id`` reference. Returns the content block to pass as
    ``legend_block``, or ``None`` (nothing to upload, client without a Files
    API, or upload failure) — callers fall back to inline bytes.

    This changes the legend's *transport only*. The legend stays ADVISORY:
    :func:`label_styles` attaches the same reconcile-don't-trust caption ahead
    of the block either way, and the system prompt's unreliable-legend framing
    is unconditional.
    """
    if not (legend_pdf or legend_image):
        return None
    upload = getattr(getattr(getattr(client, "beta", None), "files", None), "upload", None)
    if not callable(upload):
        return None
    if legend_pdf:
        name, data, media, kind = "legend.pdf", legend_pdf, "application/pdf", "document"
    else:
        name, data, media, kind = "legend.png", legend_image, "image/png", "image"
    try:
        uploaded = upload(file=(name, io.BytesIO(data), media))
    except Exception as exc:
        _log.warning("Files API legend upload failed (%s); falling back to inline bytes.", exc)
        return None
    return {"type": kind, "source": {"type": "file", "file_id": uploaded.id}}


def delete_uploaded_legend(client, legend_block: dict | None) -> None:
    """Best-effort cleanup of a block from :func:`upload_legend` (files persist
    until deleted). Never raises — a failed delete must not sink a finished run."""
    if not legend_block:
        return
    file_id = (legend_block.get("source") or {}).get("file_id")
    delete = getattr(getattr(getattr(client, "beta", None), "files", None), "delete", None)
    if not file_id or not callable(delete):
        return
    try:
        delete(file_id)
    except Exception as exc:
        _log.warning("Files API legend cleanup failed for %s (%s); the file persists.", file_id, exc)


def _labels_from_response(response) -> list[dict]:
    """The ``labels`` array from a structured-output reply.

    Scans text blocks (thinking blocks may precede the JSON) and returns the
    first parseable ``{"labels": [...]}``. Malformed output yields ``[]`` so
    every id falls back to the flagged default — never a crash, never a guess.
    """
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) != "text":
            continue
        try:
            data = json.loads(getattr(block, "text", "") or "")
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("labels"), list):
            return [item for item in data["labels"] if isinstance(item, dict)]
    return []


def label_styles(
    geometry: SheetGeometry,
    *,
    client=None,
    ppf: float | None = None,
    discipline: str = "construction",
    style_images: dict[StyleKey, bytes] | None = None,
    legend_image: bytes | None = None,
    legend_pdf: bytes | None = None,
    legend_block: dict | None = None,
    max_styles: int = 12,
    model: str | None = None,
) -> dict[StyleKey, SystemLabel]:
    """Map each candidate line style to a :class:`SystemLabel` via one LLM call.

    ``client`` is a duck-typed Anthropic client (``client.messages.create``);
    when ``None`` the vendored :func:`drawing_takeoff.client.get_client` is used.
    ``style_images`` are optional PNG bytes (rendered by
    :mod:`drawing_takeoff.geometry`); the call also works on the text stats
    alone. The advisory legend attaches as ``legend_block`` (a Files API
    reference from :func:`upload_legend`, preferred on multi-sheet sets),
    ``legend_pdf`` (single-page PDF bytes — a native ``document`` block carries
    the page's real text layer), or ``legend_image`` (PNG bytes, for raster
    legends), in that precedence. However it travels, the legend is captioned
    and prompted as ADVISORY — reconciled against the geometry, never trusted
    outright. Styles the model omits default to an ambiguous, low-confidence
    label so nothing is silently trusted.
    """
    if ppf is None:
        ppf = geometry.points_per_foot
    if model is None:
        model = api_config.REVIEW_MODEL_DEFAULT
    if client is None:
        from .client import get_client

        client = get_client()

    cands = _candidates(geometry, ppf=ppf, max_styles=max_styles)
    id_to_style = {c.style_id: c.style_key for c in cands}
    if not cands:
        return {}

    summary = (
        f"Sheet discipline: {discipline}.\n"
        f"Scale: {geometry.scale_label or 'unknown'}"
        + (f" ({ppf:g} pt/ft).\n" if ppf else ".\n")
        + f"Distinct candidate styles ({len(cands)}):\n"
        + "\n".join(_style_line(c) for c in cands)
    )

    content: list[dict] = [{"type": "text", "text": summary}]
    if style_images:
        for c in cands:
            img = style_images.get(c.style_key)
            if img:
                content.append({"type": "text", "text": f"Swatch for {c.style_id}:"})
                content.append(_image_block(img))
    if legend_block or legend_pdf or legend_image:
        content.append(
            {
                "type": "text",
                "text": "Legend / symbols list from the project's lead sheet "
                "(advisory — reconcile against the drawing, do not trust blindly):",
            }
        )
        if legend_block:
            content.append(legend_block)
        elif legend_pdf:
            content.append(_document_block(legend_pdf))
        else:
            content.append(_image_block(legend_image))
    content.append(
        {"type": "text", "text": "Return the labels JSON, with one entry for every style id above."}
    )

    response = client.messages.create(
        **_request_kwargs(
            model=model, system=_SYSTEM_PROMPT, content=content, schema=_LABELS_SCHEMA, client=client
        )
    )

    return _parse_response(response, id_to_style)


def _parse_response(response, id_to_style: dict[str, StyleKey]) -> dict[StyleKey, SystemLabel]:
    out: dict[StyleKey, SystemLabel] = {}
    for item in _labels_from_response(response):
        style = id_to_style.get(item.get("style_id"))
        if style is None:
            continue
        out[style] = SystemLabel(
            system=str(item.get("system", "")).strip() or "(unnamed)",
            measurable=bool(item.get("measurable", False)),
            confidence=str(item.get("confidence", "low")).lower(),
            ambiguous=bool(item.get("ambiguous", False)),
            reasoning=item.get("reasoning"),
        )
    # Anything the model didn't address is flagged, never silently trusted.
    for style in id_to_style.values():
        out.setdefault(
            style,
            SystemLabel(system="(unlabeled)", measurable=False, confidence="low", ambiguous=True,
                        reasoning="No label returned for this style."),
        )
    return out


# ---------------------------------------------------------------------------
# M7: label the M5 NETWORKS (generalizes label_styles), then join with M6 sizes
# ---------------------------------------------------------------------------
_NETWORK_SYSTEM_PROMPT = """\
You are a quantity-takeoff assistant for construction drawings. You are given the connected line \
NETWORKS the geometry engine found on one sheet — each network is a set of pipe runs joined end-to-end \
or by tees ("follow the line"). Each network is drawn in its own color and labeled with its id \
(N0, N1, ...) on the image, with exact geometry stats (linear feet, run count, how much of the sheet it \
spans) and the pipe SIZES the drawing itself tags along it.

The geometry and the sizes are GROUND TRUTH — never recompute or second-guess the numbers. For each \
network decide:
  - `system`: the building system it carries (e.g. "Fire-protection sprinkler", "Standpipe", \
"Domestic cold water"); use "Matchline / non-pipe" for a network that is not pipe.
  - `is_pipe`: true only if it is real pipe a length takeoff should count. A network spanning most of \
the sheet with almost no branches is likely a MATCHLINE or border (is_pipe false); a network with branch \
lines teeing off a main, or with pipe sizes tagged along it, is real pipe.
  - `confidence` (high/medium/low). Set `ambiguous` true and `confidence` low when you cannot \
confidently place it — for example a long sheet-spanning network you are unsure is a main vs a matchline. \
A human reviews every ambiguous or low-confidence network, so flag rather than guess.
  - `reasoning`: one sentence citing what you saw.

Reply with the labels JSON (the response format is enforced): a `labels` array \
with exactly one entry for every network id given."""

_NETWORK_LABELS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "labels": {
            "type": "array",
            "description": "One entry per network id given in the prompt.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "network_id": {"type": "string"},
                    "system": {
                        "type": "string",
                        "description": "System name; 'Matchline / non-pipe' for non-pipe linework.",
                    },
                    "is_pipe": {"type": "boolean", "description": "True only for real pipe to be totaled."},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "ambiguous": {
                        "type": "boolean",
                        "description": "True if you could not confidently place it; flagged for review.",
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["network_id", "system", "is_pipe", "confidence", "ambiguous", "reasoning"],
            },
        }
    },
    "required": ["labels"],
}

# Above this many numbered networks on one sheet, labeling routes to the Opus
# tier: the set-of-marks render targets 2400px on the long edge, which Opus
# 4.7+ accepts at full resolution while Sonnet 4.6 downscales to 1568px — and
# crowded marks are exactly where that resolution decides legibility. The M7
# probe validated Sonnet on sheets labeled at the default --top 8, so the
# default path stays on the cheaper model.
_DENSE_NETWORK_THRESHOLD = 12


def network_facts(geometry: SheetGeometry, networks, *, ppf: float | None = None, discipline: str = "construction") -> str:
    """Pure text the model reads beside the numbered image: per network, exact LF /
    run count / page span and the M6 size mix (the drawing's own callouts)."""
    if ppf is None:
        ppf = geometry.points_per_foot
    pw, ph = geometry.page_width_pt, geometry.page_height_pt
    lines = [
        f"Discipline: {discipline}. Scale: {geometry.scale_label or 'unknown'}"
        + (f" ({ppf:g} pt/ft)." if ppf else "."),
        f"{len(networks)} connected pipe networks, each drawn in its own color and labeled with its id "
        "on the image.",
        "Per network (geometry is exact; sizes are the drawing's own callouts snapped to each run):",
    ]
    for nw in networks:
        x0, y0, x1, y1 = nw.bbox
        pg = 100.0 * max((x1 - x0) / pw, (y1 - y0) / ph) if pw and ph else 0.0
        by = measure.linear_feet_by_size(nw.runs, geometry, ppf=ppf) if ppf else {}
        sized = ", ".join(
            f"{measure.size_label(s)}={v:.0f}LF"
            for s, v in sorted((k, v) for k, v in by.items() if k is not None)
        )
        unsized = by.get(None, 0.0)
        lf = nw.length_ft if nw.length_ft is not None else nw.length_pt
        lines.append(
            f"  {nw.id}: {lf:.0f} LF, {nw.run_count} runs, spans {pg:.0f}% of page; "
            f"sizes: {sized or '(none tagged)'}" + (f", unsized={unsized:.0f}LF" if unsized else "")
        )
    return "\n".join(lines)


def label_networks(
    geometry: SheetGeometry,
    networks,
    *,
    client=None,
    image: bytes | None = None,
    ppf: float | None = None,
    discipline: str = "construction",
    model: str | None = None,
) -> dict[str, SystemLabel]:
    """Map each network id to a :class:`SystemLabel` via one structured call.

    Generalizes :func:`label_styles` to M5 networks: the model gets the numbered
    set-of-marks ``image`` plus the per-network facts and returns, per id, the
    system and whether it's real pipe (``measurable``) — keyed on the engine's
    network ids, so its labels bind back to exact geometry. It supplies no
    numbers. Defaults to Sonnet 4.6 (M7-probe-validated at a fraction of Opus's
    cost); past :data:`_DENSE_NETWORK_THRESHOLD` networks it routes to the Opus
    tier, whose larger image cap keeps the crowded 2400px overlay legible. Pass
    ``model`` to override either way. Networks the model omits default to an
    ambiguous, non-measurable label.
    """
    networks = list(networks)
    if not networks:
        return {}
    if model is None:
        model = (
            api_config.LABELING_DENSE_MODEL_DEFAULT
            if len(networks) > _DENSE_NETWORK_THRESHOLD
            else api_config.LABELING_MODEL_DEFAULT
        )
    if client is None:
        from .client import get_client

        client = get_client()

    content: list[dict] = [{"type": "text", "text": network_facts(geometry, networks, ppf=ppf, discipline=discipline)}]
    if image:
        content.append({"type": "text", "text": "The sheet, with each network drawn in its color and labeled with its id:"})
        content.append(_image_block(image))
    content.append({"type": "text", "text": "Return the labels JSON, with one entry for every network id above."})

    response = client.messages.create(
        **_request_kwargs(
            model=model, system=_NETWORK_SYSTEM_PROMPT, content=content,
            schema=_NETWORK_LABELS_SCHEMA, client=client,
        )
    )
    return _parse_network_response(response, networks)


def _parse_network_response(response, networks) -> dict[str, SystemLabel]:
    ids = {nw.id for nw in networks}
    out: dict[str, SystemLabel] = {}
    for item in _labels_from_response(response):
        nid = item.get("network_id")
        if nid not in ids:
            continue
        out[nid] = SystemLabel(
            system=str(item.get("system", "")).strip() or "(unnamed)",
            measurable=bool(item.get("is_pipe", False)),
            confidence=str(item.get("confidence", "low")).lower(),
            ambiguous=bool(item.get("ambiguous", False)),
            reasoning=item.get("reasoning"),
        )
    for nw in networks:
        out.setdefault(
            nw.id,
            SystemLabel(system="(unlabeled)", measurable=False, confidence="low", ambiguous=True,
                        reasoning="No label returned for this network."),
        )
    return out


# ---------------------------------------------------------------------------
# Second look: re-check flagged networks from high-DPI close-ups
# ---------------------------------------------------------------------------
_SECOND_LOOK_SYSTEM_PROMPT = """\
You are a quantity-takeoff assistant for construction drawings. A first pass over the whole \
sheet could not confidently place the pipe NETWORKS listed here — each was flagged for human \
review. You now get a high-resolution CLOSE-UP of each flagged network (highlighted in color \
and labeled with its id) showing detail the whole-sheet view blurs: tees and branch \
connections, fittings, and nearby callouts. The first-pass label and reasoning are given as \
context; confirm or overturn them based on what the close-up actually shows.

The geometry stats are GROUND TRUTH — never recompute or second-guess the numbers. The same \
rules as the first pass apply: branch lines teeing off a main or pipe sizes tagged along the \
run mean real pipe; a sheet-spanning line with no branches is likely a matchline or border \
(`is_pipe` false). If the close-up still does not settle it, keep `ambiguous` true and \
`confidence` low — a human reviews every flagged network; never guess.

Reply with the labels JSON (the response format is enforced): a `labels` array \
with exactly one entry for every network id given."""


def needs_second_look(label: SystemLabel) -> bool:
    """Review-list membership — what the close-up re-check should target.

    Ambiguous either way, or measurable but below the trust bar. A confident
    non-pipe exclusion (e.g. a high-confidence matchline) is settled, not
    flagged, so it is not re-checked.
    """
    return label.ambiguous or (label.measurable and not label.trusted)


def second_look_networks(
    geometry: SheetGeometry,
    networks,
    first_pass: dict[str, SystemLabel],
    crops: dict[str, bytes],
    *,
    client=None,
    ppf: float | None = None,
    discipline: str = "construction",
    model: str | None = None,
) -> dict[str, SystemLabel]:
    """Re-label flagged networks from high-DPI close-ups, in one focused call.

    ``networks`` is the flagged subset (see :func:`needs_second_look`) and
    ``crops`` maps network id -> close-up PNG from
    :func:`drawing_takeoff.geometry.render_network_crop_png` — the surgical
    replacement for blanket sheet tiling: the engine knows exactly where each
    ambiguity lives, so only that region is rendered at high DPI. Defaults to
    the escalation model (flagged cases are by definition the hard ones).
    Returns labels for the given ids only — callers merge over the first pass
    with ``labels.update(...)``. Reasoning is prefixed ``second look:`` so the
    review surfaces show which pass produced each label, and a network the
    model omits (or that has no crop) keeps a flagged default — the re-check
    can only refine the review list, never silently widen trust.
    """
    networks = [nw for nw in networks if nw.id in crops]
    if not networks:
        return {}
    if model is None:
        model = api_config.LABELING_ESCALATION_MODEL_DEFAULT
    if client is None:
        from .client import get_client

        client = get_client()

    content: list[dict] = [{"type": "text", "text": network_facts(geometry, networks, ppf=ppf, discipline=discipline)}]
    notes = []
    for nw in networks:
        lab = first_pass.get(nw.id)
        if lab is not None:
            notes.append(
                f"  {nw.id}: first pass said {lab.system!r} (is_pipe={lab.measurable}, "
                f"confidence={lab.confidence}, ambiguous={lab.ambiguous})"
                + (f" — {lab.reasoning}" if lab.reasoning else "")
            )
    if notes:
        content.append({"type": "text", "text": "First-pass labels (flagged for review):\n" + "\n".join(notes)})
    for nw in networks:
        content.append({"type": "text", "text": f"Close-up of {nw.id}:"})
        content.append(_image_block(crops[nw.id]))
    content.append({"type": "text", "text": "Return the labels JSON, with one entry for every network id above."})

    response = client.messages.create(
        **_request_kwargs(
            model=model, system=_SECOND_LOOK_SYSTEM_PROMPT, content=content,
            schema=_NETWORK_LABELS_SCHEMA, client=client,
        )
    )
    out = _parse_network_response(response, networks)
    return {
        nid: replace(lab, reasoning=f"second look: {lab.reasoning}" if lab.reasoning else "second look")
        for nid, lab in out.items()
    }


# ---------------------------------------------------------------------------
# M10: name the M9 symbol clusters (counts takeoff) — engine counts, model names
# ---------------------------------------------------------------------------
_SYMBOL_SYSTEM_PROMPT = """\
You are a quantity-takeoff assistant for construction drawings. The geometry engine found the \
REPEATED SYMBOLS on one sheet by exact congruence: each cluster is one distinct drawn shape, \
repeated `count` times (rotated and mirrored placements already merged). You get one close-up \
CROP of one exemplar instance per cluster — the symbol is outlined in red; surrounding linework \
is context, not part of it — plus exact stats (count, drawn size at sheet scale, page spread).

The counts and sizes are GROUND TRUTH — never recount, and supply no numbers. For each cluster \
decide:
  - `component`: what the symbol IS (e.g. "Sprinkler head (pendent)", "Water closet", \
"Lavatory", "Supply diffuser", "Floor drain"). Use any legend given, reconciled against what \
the crop actually shows — legends are advisory, frequently incomplete or wrong.
  - `countable`: true only for a discrete physical component an EA count takeoff should total \
(plumbing fixture, sprinkler head, diffuser, device, equipment). Door swings, hatch/tile \
pattern, text or outlined lettering, size/length labels and their background masks, grid \
bubbles, north arrows, section/detail markers, dimension ticks and other annotation are NOT \
countable.
  - `confidence` (high/medium/low) and `ambiguous`: when you cannot confidently name it, set \
`ambiguous` true and `confidence` low — a human reviews flagged clusters; never guess.
  - `reasoning`: one sentence citing what you saw.

Reply with the labels JSON (the response format is enforced): a `labels` array \
with exactly one entry for every symbol id given."""

_SYMBOL_LABELS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "labels": {
            "type": "array",
            "description": "One entry per symbol id given in the prompt.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "symbol_id": {"type": "string"},
                    "component": {
                        "type": "string",
                        "description": "What the symbol is; name annotation for what it is too.",
                    },
                    "countable": {
                        "type": "boolean",
                        "description": "True only for a discrete physical component to total (EA).",
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "ambiguous": {
                        "type": "boolean",
                        "description": "True if you could not confidently name it; flagged for review.",
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["symbol_id", "component", "countable", "confidence", "ambiguous", "reasoning"],
            },
        }
    },
    "required": ["labels"],
}


def symbol_facts(geometry: SheetGeometry, clusters, *, ppf: float | None = None,
                 discipline: str = "construction") -> str:
    """Pure text the model reads beside the exemplar crops: per cluster, the
    exact count / drawn size / parts / drawn-variant count / page spread.
    Every number here comes from the engine."""
    if ppf is None:
        ppf = geometry.points_per_foot
    pw, ph = geometry.page_width_pt, geometry.page_height_pt
    lines = [
        f"Discipline: {discipline}. Scale: {geometry.scale_label or 'unknown'}"
        + (f" ({ppf:g} pt/ft)." if ppf else "."),
        f"{len(clusters)} distinct repeated symbols (congruence clusters); counts are exact.",
        "Per cluster:",
    ]
    for c in clusters:
        if ppf:
            size = f"{c.width_pt / ppf:.1f} x {c.height_pt / ppf:.1f} ft"
        else:
            size = f"{c.width_pt:.0f} x {c.height_pt:.0f} pt"
        x0, y0, x1, y1 = c.bbox
        pg = 100.0 * max((x1 - x0) / pw, (y1 - y0) / ph) if pw and ph else 0.0
        lines.append(
            f"  {c.id}: {c.count} instances, drawn size {size}, {c.paths_per_instance} path(s) "
            f"per instance, {c.variants} drawn variant(s), instances spread over {pg:.0f}% of page"
        )
    return "\n".join(lines)


def label_symbols(
    geometry: SheetGeometry,
    clusters,
    *,
    client=None,
    crops: dict[str, bytes] | None = None,
    ppf: float | None = None,
    discipline: str = "construction",
    legend_image: bytes | None = None,
    legend_pdf: bytes | None = None,
    legend_block: dict | None = None,
    model: str | None = None,
) -> dict[str, SystemLabel]:
    """Name each symbol cluster via one structured call — the M10 step.

    Generalizes :func:`label_networks` to M9 clusters: the model gets one
    highlighted exemplar crop per cluster plus the engine's facts and returns,
    per id, the component and whether it is countable (mapped onto
    :class:`SystemLabel`: ``system`` = component, ``measurable`` = countable) —
    keyed on the engine's cluster ids, so labels bind back to exact counts. It
    supplies no numbers. Clusters the model omits default to an ambiguous,
    non-countable label; the advisory legend rides the same transport (and
    framing) as everywhere else.
    """
    clusters = list(clusters)
    if not clusters:
        return {}
    if ppf is None:
        ppf = geometry.points_per_foot
    if model is None:
        model = api_config.LABELING_MODEL_DEFAULT
    if client is None:
        from .client import get_client

        client = get_client()

    content: list[dict] = [
        {"type": "text", "text": symbol_facts(geometry, clusters, ppf=ppf, discipline=discipline)}
    ]
    for c in clusters:
        img = (crops or {}).get(c.id)
        if img:
            content.append({"type": "text", "text": f"Close-up of {c.id} (the symbol is outlined in red):"})
            content.append(_image_block(img))
    if legend_block or legend_pdf or legend_image:
        content.append(
            {
                "type": "text",
                "text": "Legend / symbols list from the project's lead sheet "
                "(advisory — reconcile against the crops, do not trust blindly):",
            }
        )
        if legend_block:
            content.append(legend_block)
        elif legend_pdf:
            content.append(_document_block(legend_pdf))
        else:
            content.append(_image_block(legend_image))
    content.append(
        {"type": "text", "text": "Return the labels JSON, with one entry for every symbol id above."}
    )

    response = client.messages.create(
        **_request_kwargs(
            model=model, system=_SYMBOL_SYSTEM_PROMPT, content=content,
            schema=_SYMBOL_LABELS_SCHEMA, client=client,
        )
    )
    return _parse_symbol_response(response, clusters)


def _parse_symbol_response(response, clusters) -> dict[str, SystemLabel]:
    ids = {c.id for c in clusters}
    out: dict[str, SystemLabel] = {}
    for item in _labels_from_response(response):
        sid = item.get("symbol_id")
        if sid not in ids:
            continue
        out[sid] = SystemLabel(
            system=str(item.get("component", "")).strip() or "(unnamed)",
            measurable=bool(item.get("countable", False)),
            confidence=str(item.get("confidence", "low")).lower(),
            ambiguous=bool(item.get("ambiguous", False)),
            reasoning=item.get("reasoning"),
        )
    for c in clusters:
        out.setdefault(
            c.id,
            SystemLabel(system="(unlabeled)", measurable=False, confidence="low", ambiguous=True,
                        reasoning="No label returned for this cluster."),
        )
    return out


_SYMBOL_SECOND_LOOK_PROMPT = """\
You are a quantity-takeoff assistant for construction drawings. A first pass could not \
confidently name the repeated SYMBOLS listed here — each was flagged for human review. You now \
get wider, high-resolution close-ups of up to three DIFFERENT instances of each flagged cluster \
(the symbol outlined in red; the surroundings are context — what the symbol sits on or connects \
to is often what identifies it). The first-pass label and reasoning are given; confirm or \
overturn them based on what the close-ups actually show.

The counts are GROUND TRUTH — never recount, and supply no numbers. The same rules apply: \
`countable` only for a discrete physical component (fixture / head / diffuser / device); \
annotation, text, masks and pattern are not countable. If the close-ups still do not settle \
it, keep `ambiguous` true and `confidence` low — a human reviews every flagged cluster; never \
guess.

Reply with the labels JSON (the response format is enforced): a `labels` array \
with exactly one entry for every symbol id given."""


def second_look_symbols(
    geometry: SheetGeometry,
    clusters,
    first_pass: dict[str, SystemLabel],
    crops: dict[str, list[bytes]],
    *,
    client=None,
    ppf: float | None = None,
    discipline: str = "construction",
    model: str | None = None,
) -> dict[str, SystemLabel]:
    """Re-name flagged clusters from wider multi-instance close-ups, in one call.

    ``clusters`` is the flagged subset (see :func:`needs_second_look`) and
    ``crops`` maps cluster id -> up to three close-up PNGs of different
    instances — context disambiguates (the same oval reads as a lavatory on a
    counter run and a urinal on a wall), which is why the second look widens
    the margin and shows several placements instead of re-reading one. Defaults
    to the escalation model. Returns labels for the given ids only — callers
    merge with ``labels.update(...)``; reasoning is prefixed ``second look:``
    and an omitted cluster keeps its flagged default, so the re-check can only
    refine the review list, never silently widen trust.
    """
    clusters = [c for c in clusters if crops.get(c.id)]
    if not clusters:
        return {}
    if model is None:
        model = api_config.LABELING_ESCALATION_MODEL_DEFAULT
    if client is None:
        from .client import get_client

        client = get_client()

    content: list[dict] = [
        {"type": "text", "text": symbol_facts(geometry, clusters, ppf=ppf, discipline=discipline)}
    ]
    notes = []
    for c in clusters:
        lab = first_pass.get(c.id)
        if lab is not None:
            notes.append(
                f"  {c.id}: first pass said {lab.system!r} (countable={lab.measurable}, "
                f"confidence={lab.confidence}, ambiguous={lab.ambiguous})"
                + (f" — {lab.reasoning}" if lab.reasoning else "")
            )
    if notes:
        content.append({"type": "text", "text": "First-pass labels (flagged for review):\n" + "\n".join(notes)})
    for c in clusters:
        for i, img in enumerate(crops[c.id][:3]):
            content.append({"type": "text", "text": f"Instance {i + 1} of {c.id}:"})
            content.append(_image_block(img))
    content.append({"type": "text", "text": "Return the labels JSON, with one entry for every symbol id above."})

    response = client.messages.create(
        **_request_kwargs(
            model=model, system=_SYMBOL_SECOND_LOOK_PROMPT, content=content,
            schema=_SYMBOL_LABELS_SCHEMA, client=client,
        )
    )
    out = _parse_symbol_response(response, clusters)
    return {
        sid: replace(lab, reasoning=f"second look: {lab.reasoning}" if lab.reasoning else "second look")
        for sid, lab in out.items()
    }


def pipe_runs_from_style_labels(runs_by, style_labels):
    """Union of runs from every style the LLM is **confident** is pipe (``trusted``
    = measurable, not ambiguous, decent confidence) — the candidate set fed to
    :func:`drawing_takeoff.measure.networks`. Style-spanning on purpose: mains and
    branches in different pens all join in, replacing the single heaviest-dark
    proxy. Styles it calls measurable-but-ambiguous are deliberately left OUT and
    surfaced for confirmation rather than silently inflating the total."""
    pipe = []
    for style, rs in runs_by.items():
        lab = style_labels.get(style)
        if lab is not None and lab.trusted:
            pipe.extend(rs)
    return pipe


_PAGE_SPAN_ADVISORY = 0.80


def _page_spanning_confirms(networks, labels: dict[str, SystemLabel], geometry: SheetGeometry) -> list[str]:
    """Counted networks that span ~the whole sheet — flagged for a main-vs-matchline
    glance (a long main and a matchline both span the sheet). Shared by the text
    report and the workbook so the two review surfaces can't drift."""
    pw, ph = geometry.page_width_pt, geometry.page_height_pt
    out = []
    for nw in networks:
        lab = labels.get(nw.id)
        if lab is not None and lab.trusted and pw and ph:
            x0, y0, x1, y1 = nw.bbox
            if max((x1 - x0) / pw, (y1 - y0) / ph) >= _PAGE_SPAN_ADVISORY:
                out.append(f"{nw.id} ({lab.system})")
    return out


def system_size_takeoff(
    networks,
    labels: dict[str, SystemLabel],
    geometry: SheetGeometry,
    *,
    ppf: float,
    radius_ft: float = measure._DEFAULT_SIZE_RADIUS_FT,
) -> tuple[dict[tuple[str, str], float], list[str]]:
    """Join network->system (M7) with run->size (M6) into a System x Size LF table.

    Only **trusted** pipe networks (measurable, not ambiguous, confident) roll
    into the totals; a non-pipe (matchline) network is excluded with a note, and
    an ambiguous / low-confidence one is routed to the review list rather than
    counted. Returns ``(by_system_size, review_notes)``. The LLM supplied only
    labels, so every LF here traces back to exact geometry.
    """
    by: dict[tuple[str, str], float] = {}
    review: list[str] = []
    for nw in networks:
        lf = nw.length_ft if nw.length_ft is not None else (nw.length_pt / ppf)
        label = labels.get(nw.id)
        if label is None or not label.measurable:
            why = "no label returned" if label is None else f"non-pipe ({label.system})"
            review.append(
                f"{nw.id}: excluded — {why} — {lf:,.0f} LF"
                + (f" [{label.reasoning}]" if label is not None and label.reasoning else "")
            )
            continue
        if not label.trusted:
            review.append(
                f"{nw.id}: {label.system} flagged "
                f"({label.confidence}{', ambiguous' if label.ambiguous else ''}) — {lf:,.0f} LF"
            )
            continue
        for size, slf in measure.linear_feet_by_size(nw.runs, geometry, ppf=ppf, radius_ft=radius_ft).items():
            key = (label.system, measure.size_label(size) if size is not None else "unsized")
            by[key] = by.get(key, 0.0) + slf
    return by, review


def build_system_size_report(
    networks,
    labels: dict[str, SystemLabel],
    geometry: SheetGeometry,
    *,
    ppf: float | None = None,
    radius_ft: float = measure._DEFAULT_SIZE_RADIUS_FT,
) -> str:
    """M7 deliverable: the System x Size linear-feet table + a review list."""
    if ppf is None:
        ppf = geometry.points_per_foot
    head = f"=== M7 takeoff: {geometry.ref.source} (page {geometry.ref.page_index}) ==="
    if not ppf:
        return head + "\n  no scale on sheet; cannot compute footage"

    by, review = system_size_takeoff(networks, labels, geometry, ppf=ppf, radius_ft=radius_ft)
    systems: dict[str, dict[str, float]] = {}
    for (system, size), lf in by.items():
        systems.setdefault(system, {})[size] = lf

    lines = [head, "", "TAKEOFF — linear feet by system and size (trusted pipe networks):"]
    if systems:
        label_to_in = {v: k for k, v in measure._SIZE_LABEL.items()}
        for system in sorted(systems, key=lambda s: -sum(systems[s].values())):
            lines.append(f"  {system}: {sum(systems[system].values()):,.1f} LF")
            for size, lf in sorted(systems[system].items(), key=lambda kv: label_to_in.get(kv[0], 1e9)):
                lines.append(f"      {size:>10s}  {lf:>9,.1f} LF")
    else:
        lines.append("  (no trusted pipe networks)")

    # Counted networks that span ~the whole sheet are worth a human glance (a long
    # main and a matchline both span the sheet; the model judged this one pipe).
    spanning = _page_spanning_confirms(networks, labels, geometry)
    if spanning:
        lines += ["", "CONFIRM (counted, but span ~the whole sheet — main vs matchline): " + ", ".join(spanning)]
    if review:
        lines += ["", "REVIEW (not counted):"] + [f"  {r}" for r in review]
    return "\n".join(lines)


def takeoff_tables(networks, labels: dict[str, SystemLabel], geometry: SheetGeometry, *,
                   ppf: float | None = None, radius_ft: float = measure._DEFAULT_SIZE_RADIUS_FT) -> dict:
    """Assemble what the M8 outputs need from networks + labels, as **plain data**
    (so the Excel / PDF writers never touch the engine's model types):

      ``by_system_size`` — ``{(system, size_label): LF}`` (trusted pipe only),
      ``detail``         — one dict per network (id, system, is_pipe, counted, …),
      ``review``         — the not-counted / confirm notes.
    """
    if ppf is None:
        ppf = geometry.points_per_foot
    by, review = system_size_takeoff(networks, labels, geometry, ppf=ppf, radius_ft=radius_ft)
    pw, ph = geometry.page_width_pt, geometry.page_height_pt
    detail = []
    for nw in networks:
        lab = labels.get(nw.id)
        x0, y0, x1, y1 = nw.bbox
        pct = round(100.0 * max((x1 - x0) / pw, (y1 - y0) / ph)) if pw and ph else 0
        by_size = measure.linear_feet_by_size(nw.runs, geometry, ppf=ppf, radius_ft=radius_ft)
        size_parts = [f"{measure.size_label(s)}={v:.0f}" for s, v in sorted((k, v) for k, v in by_size.items() if k is not None)]
        if by_size.get(None, 0.0):
            size_parts.append(f"unsized={by_size[None]:.0f}")  # keep unsized so Sizes reconciles to LF
        sizes = ", ".join(size_parts)
        detail.append({
            "network": nw.id,
            "system": lab.system if lab is not None else "(unlabeled)",
            "is_pipe": bool(lab is not None and lab.measurable),
            "counted": bool(lab is not None and lab.trusted),
            "confidence": lab.confidence if lab is not None else "low",
            "ambiguous": bool(lab is not None and lab.ambiguous),
            "linear_feet": round(nw.length_ft if nw.length_ft is not None else nw.length_pt / ppf, 1),
            "pct_page": pct,
            "sizes": sizes,
            "reasoning": (lab.reasoning if lab is not None else "") or "",
        })
    review_notes = list(review)
    spanning = _page_spanning_confirms(networks, labels, geometry)
    if spanning:
        review_notes.append(
            "CONFIRM (counted, but span ~the whole sheet — main vs matchline): " + ", ".join(spanning)
        )
    return {"by_system_size": by, "detail": detail, "review": review_notes}


def _load_legend_attachment(path: str, page: int) -> tuple[str, bytes]:
    """Load the advisory legend as ``("pdf", bytes)`` or ``("image", bytes)``.

    A PDF legend page is extracted as a standalone single-page PDF and sent as
    a native ``document`` block, so the model reads the page's real text layer
    instead of OCR-ing a raster. Raster legends are re-encoded to PNG — the
    image block advertises ``image/png``, so a ``.jpg``/``.webp`` legend must
    not pass through with a mislabeled media type.
    """
    from .geometry import extract_page_pdf, image_file_to_png

    if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")):
        return "image", image_file_to_png(path)
    return "pdf", extract_page_pdf(path, page)


def main(argv: list[str] | None = None) -> int:
    """CLI: label a sheet's line styles and roll trusted ones up by system.

    The first command that calls Claude — it needs ``ANTHROPIC_API_KEY``. The
    legend lives on a separate lead sheet on most drawing sets, so pass it with
    ``--legend`` (advisory); without it the model labels from the geometry stats
    and rendered swatches alone.

      python -m drawing_takeoff.legend SHEET.pdf [--legend LEAD.pdf] \\
             [--discipline "fire protection"]
    """
    import argparse
    import os
    import sys
    from collections import defaultdict

    ap = argparse.ArgumentParser(description="M3 legend labeling (style -> system) via one Claude call.")
    ap.add_argument("pdf")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--legend", default=None, help="lead-sheet PDF or image with the legend/symbols (advisory)")
    ap.add_argument("--legend-page", type=int, default=0)
    ap.add_argument("--discipline", default="construction", help='e.g. "fire protection", "plumbing", "HVAC"')
    ap.add_argument("--max-styles", type=int, default=12)
    ap.add_argument("--system-size", action="store_true",
                    help="M7: label the connected networks and print the System x Size takeoff")
    ap.add_argument("--top", type=int, default=8, help="largest N networks to label (with --system-size)")
    ap.add_argument("--no-second-look", action="store_true",
                    help="skip the high-DPI close-up re-check of flagged networks (with --system-size)")
    ap.add_argument("--out", default=None, help="with --system-size: also write takeoff.xlsx + a marked-up PDF here")
    args = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("ANTHROPIC_API_KEY is not set — M3 is the LLM step and needs a key.", file=sys.stderr)
        return 2

    from . import measure
    from .geometry import extract_pdf_geometry, render_style_swatch

    geom = extract_pdf_geometry(args.pdf, pages=[args.page])[0]

    if args.system_size:
        # The System×Size path (style→pipe → networks → size → system, second-look
        # re-check, Excel + marked-up PDF) lives in the pipeline so the GUI and this
        # CLI run one orchestration. Deferred import avoids a legend<->pipeline cycle.
        from . import pipeline

        legend_kind, legend_bytes = (
            _load_legend_attachment(args.legend, args.legend_page) if args.legend else (None, None)
        )
        sheet = pipeline.system_size_for_sheet(
            geom, discipline=args.discipline, max_styles=args.max_styles, top=args.top,
            second_look=not args.no_second_look,
            legend_pdf=legend_bytes if legend_kind == "pdf" else None,
            legend_image=legend_bytes if legend_kind == "image" else None,
        )
        print(sheet.report)

        if args.out:
            result = pipeline.SystemSizeResult(sheet_count=1)
            pipeline._absorb_sheet(result, sheet)
            folder = pipeline.write_system_size_export(result, args.out, project_name="takeoff")
            print(f"\nWrote {folder}/  (takeoff.xlsx + one marked-up PDF)")
        return 0

    ppf = geom.points_per_foot
    cands = _candidates(geom, ppf=ppf, max_styles=args.max_styles)
    if not cands:
        print("No measurable line styles found on this sheet.")
        return 0

    swatches = {c.style_key: render_style_swatch(c.style_key) for c in cands}
    legend_kind, legend_bytes = (
        _load_legend_attachment(args.legend, args.legend_page) if args.legend else (None, None)
    )

    labels = label_styles(
        geom,
        ppf=ppf,
        discipline=args.discipline,
        style_images=swatches,
        legend_image=legend_bytes if legend_kind == "image" else None,
        legend_pdf=legend_bytes if legend_kind == "pdf" else None,
        max_styles=args.max_styles,
    )
    feet = measure.linear_feet_by_style(geom, ppf) if ppf else {}

    print(f"=== M3 legend labels: {geom.ref.source} (page {geom.ref.page_index}) ===")
    print(f"scale: {geom.scale_label or 'unknown'}" + (f"  ->  {ppf:g} pt/ft" if ppf else ""))
    if args.legend:
        print(f"legend image: {args.legend} (advisory)")
    print("\nper-style labels:")
    for c in cands:
        lab = labels[c.style_key]
        lf = feet.get(c.style_key)
        lfs = f"{lf:,.0f} LF" if lf else "-"
        flag = "  [AMBIGUOUS — review]" if lab.ambiguous else ""
        print(f"  {c.style_id} {lfs:>10}  measurable={str(lab.measurable):5} {lab.confidence:>6} -> {lab.system}{flag}")
        if lab.reasoning:
            print(f"        {lab.reasoning}")

    # Roll TRUSTED styles up by system (SystemLabel.trusted = measurable, not
    # ambiguous, decent confidence); flag any measurable-but-not-trusted for
    # review so a low-confidence style is never silently counted.
    systems: dict[str, float] = defaultdict(float)
    flagged = []
    for c in cands:
        lab = labels[c.style_key]
        lf = feet.get(c.style_key)
        if lab.trusted:
            systems[lab.system] += (lf or 0.0)
        elif lab.measurable:
            flagged.append((c, lab, lf))

    if ppf:
        print("\nTAKEOFF by system (trusted styles — measurable, not ambiguous, confident):")
        if systems:
            for name, lf in sorted(systems.items(), key=lambda kv: -kv[1]):
                print(f"  {name}: {lf:,.0f} LF")
        else:
            print("  (nothing met the trust bar — see flagged styles)")
    else:
        # No scale -> LF is unknowable; report that instead of a false "0 LF".
        print("\nscale not detected — linear footage NOT computed (style labels above stand).")
        if systems:
            print("  trusted measurable systems:", ", ".join(sorted(systems)))

    if flagged:
        print("\nFLAGGED for your review (measurable but ambiguous or low-confidence — NOT counted):")
        for c, lab, lf in flagged:
            lfs = f"{lf:,.0f} LF" if lf else "LF n/a"
            print(
                f"  {c.style_id} ({lfs}): {lab.system} "
                f"[conf={lab.confidence}, ambiguous={lab.ambiguous}] — {lab.reasoning or ''}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
