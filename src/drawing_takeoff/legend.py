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

Structured output is obtained via a forced single tool call
(``record_system_labels``); the response is parsed from the ``tool_use`` block.
No PyMuPDF here — images arrive as PNG bytes (rendered by ``geometry``), and the
Anthropic client is duck-typed and injectable for hermetic tests.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from . import measure
from .core import api_config
from .models import SheetGeometry, StyleKey, SystemLabel

_log = logging.getLogger(__name__)

_MAX_TOKENS = 4096  # the structured reply is small; well under the streaming threshold

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

Call the `record_system_labels` tool exactly once, with one entry for every \
style id provided."""

_TOOL = {
    "name": "record_system_labels",
    "description": "Record the building system and measurability of each line style on the sheet.",
    "strict": True,
    "input_schema": {
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
    },
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


def label_styles(
    geometry: SheetGeometry,
    *,
    client=None,
    ppf: float | None = None,
    discipline: str = "construction",
    style_images: dict[StyleKey, bytes] | None = None,
    legend_image: bytes | None = None,
    max_styles: int = 12,
    model: str | None = None,
) -> dict[StyleKey, SystemLabel]:
    """Map each candidate line style to a :class:`SystemLabel` via one LLM call.

    ``client`` is a duck-typed Anthropic client (``client.messages.create``);
    when ``None`` the vendored :func:`drawing_takeoff.client.get_client` is used.
    ``style_images`` / ``legend_image`` are optional PNG bytes (rendered by
    :mod:`drawing_takeoff.geometry`); the call also works on the text stats
    alone. Styles the model omits default to an ambiguous, low-confidence label
    so nothing is silently trusted.
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
    if legend_image:
        content.append(
            {
                "type": "text",
                "text": "Legend / symbols list from the project's lead sheet "
                "(advisory — reconcile against the drawing, do not trust blindly):",
            }
        )
        content.append(_image_block(legend_image))
    content.append(
        {"type": "text", "text": "Call record_system_labels once with an entry for every style id above."}
    )

    response = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "record_system_labels"},
    )

    return _parse_response(response, id_to_style)


def _parse_response(response, id_to_style: dict[str, StyleKey]) -> dict[StyleKey, SystemLabel]:
    out: dict[StyleKey, SystemLabel] = {}
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) != "tool_use" or getattr(block, "name", None) != "record_system_labels":
            continue
        for item in (getattr(block, "input", {}) or {}).get("labels", []):
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

Call `record_network_labels` exactly once, with one entry for every network id given."""

_NETWORK_TOOL = {
    "name": "record_network_labels",
    "description": "Record the building system and measurability of each numbered network on the sheet.",
    "strict": True,
    "input_schema": {
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
    },
}


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
    discipline: str = "construction",
    model: str | None = None,
) -> dict[str, SystemLabel]:
    """Map each network id to a :class:`SystemLabel` via one forced tool call.

    Generalizes :func:`label_styles` to M5 networks: the model gets the numbered
    set-of-marks ``image`` plus the per-network facts and returns, per id, the
    system and whether it's real pipe (``measurable``) — keyed on the engine's
    network ids, so its labels bind back to exact geometry. It supplies no
    numbers. Defaults to Sonnet 4.6, which the M7 probe validated for this
    vision+reasoning task at a fraction of Opus's cost; pass ``model`` to change.
    Networks the model omits default to an ambiguous, non-measurable label.
    """
    networks = list(networks)
    if not networks:
        return {}
    if model is None:
        model = api_config.MODEL_SONNET_46
    if client is None:
        from .client import get_client

        client = get_client()

    content: list[dict] = [{"type": "text", "text": network_facts(geometry, networks, discipline=discipline)}]
    if image:
        content.append({"type": "text", "text": "The sheet, with each network drawn in its color and labeled with its id:"})
        content.append(_image_block(image))
    content.append({"type": "text", "text": "Call record_network_labels once, with one entry for every network id above."})

    response = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=_NETWORK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        tools=[_NETWORK_TOOL],
        tool_choice={"type": "tool", "name": "record_network_labels"},
    )
    return _parse_network_response(response, networks)


def _parse_network_response(response, networks) -> dict[str, SystemLabel]:
    ids = {nw.id for nw in networks}
    out: dict[str, SystemLabel] = {}
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) != "tool_use" or getattr(block, "name", None) != "record_network_labels":
            continue
        for item in (getattr(block, "input", {}) or {}).get("labels", []):
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


_PAGE_SPAN_ADVISORY = 0.80


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
    pw, ph = geometry.page_width_pt, geometry.page_height_pt
    spanning = []
    for nw in networks:
        lab = labels.get(nw.id)
        if lab is not None and lab.trusted and pw and ph:
            x0, y0, x1, y1 = nw.bbox
            if max((x1 - x0) / pw, (y1 - y0) / ph) >= _PAGE_SPAN_ADVISORY:
                spanning.append(f"{nw.id} ({lab.system})")
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
        sizes = ", ".join(
            f"{measure.size_label(s)}={v:.0f}" for s, v in sorted((k, v) for k, v in by_size.items() if k is not None)
        )
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
    return {"by_system_size": by, "detail": detail, "review": list(review)}


def _load_legend_image(path: str, page: int) -> bytes:
    """Read a legend image (any common raster) or rasterize a legend PDF page.

    Always returns PNG bytes — the request advertises ``image/png`` for every
    image block, so a ``.jpg``/``.webp`` legend must be re-encoded, not passed
    through with a mislabeled media type.
    """
    from .geometry import image_file_to_png, render_page_png

    if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")):
        return image_file_to_png(path)
    return render_page_png(path, page, dpi=150)


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
    ap.add_argument("--out", default=None, help="with --system-size: also write takeoff.xlsx + a marked-up PDF here")
    args = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("ANTHROPIC_API_KEY is not set — M3 is the LLM step and needs a key.", file=sys.stderr)
        return 2

    from . import measure
    from .geometry import extract_pdf_geometry, render_style_swatch

    geom = extract_pdf_geometry(args.pdf, pages=[args.page])[0]

    if args.system_size:
        from .geometry import render_networks_png

        ppf = geom.points_per_foot
        runs_by = measure.runs_by_style(geom, ppf=ppf, exclude_border=True)
        pipe_style = measure.heaviest_dark_style(k for k, rs in runs_by.items() if rs)
        pipe = runs_by.get(pipe_style, []) if pipe_style is not None else []
        all_nets = measure.networks(pipe, ppf=ppf)
        nets = all_nets[: args.top]
        if not nets:
            print("No pipe networks found on this sheet.")
            return 0
        image = render_networks_png(args.pdf, args.page, nets)
        labels = label_networks(geom, nets, image=image, discipline=args.discipline)

        # Footage outside the labeled set — surfaced, never silently dropped.
        extra: list[str] = []
        remainder = all_nets[args.top:]
        if remainder:
            extra.append(
                f"NOT LABELED: {len(remainder)} smaller networks beyond top {args.top} = "
                f"{sum((nw.length_ft or 0.0) for nw in remainder):,.0f} LF (raise --top to include them)."
            )
        other_n = other_lf = 0
        for k, rs in runs_by.items():
            if rs and k != pipe_style and k.stroke_color is not None and max(k.stroke_color) < 0.30 and (k.width or 0) > 0:
                other_n += 1
                other_lf += sum(r.length_pt for r in rs) / ppf
        if other_n:
            extra.append(
                f"OTHER DARK LINEWEIGHTS not in this takeoff: {other_n} style(s) = {other_lf:,.0f} LF "
                "(candidate is the heaviest-dark pen; confirm none is pipe — multi-style selection is future work)."
            )

        report = build_system_size_report(nets, labels, geom, ppf=ppf)
        if extra:
            report += "\n\n" + "\n".join(extra)
        print(report)

        if args.out:
            import datetime as _dt
            from pathlib import Path as _Path

            from . import export
            from .geometry import write_marked_up_pdf

            tables = takeoff_tables(nets, labels, geom, ppf=ppf)
            tables["review"] = tables["review"] + extra
            folder = _Path(args.out) / f"takeoff_{_dt.datetime.now():%Y%m%d_%H%M%S}"
            folder.mkdir(parents=True, exist_ok=True)
            export.build_takeoff_workbook(tables).save(folder / "takeoff.xlsx")
            write_marked_up_pdf(args.pdf, args.page, nets, str(folder / "takeoff_markup.pdf"))
            print(f"\nWrote {folder}/  (takeoff.xlsx, takeoff_markup.pdf)")
        return 0

    ppf = geom.points_per_foot
    cands = _candidates(geom, ppf=ppf, max_styles=args.max_styles)
    if not cands:
        print("No measurable line styles found on this sheet.")
        return 0

    swatches = {c.style_key: render_style_swatch(c.style_key) for c in cands}
    legend_image = _load_legend_image(args.legend, args.legend_page) if args.legend else None

    labels = label_styles(
        geom,
        ppf=ppf,
        discipline=args.discipline,
        style_images=swatches,
        legend_image=legend_image,
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
