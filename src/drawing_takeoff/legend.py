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
    """The most takeoff-relevant styles, ranked by total measured length."""
    runs_by_style = measure.runs_by_style(geometry, ppf=ppf)
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
