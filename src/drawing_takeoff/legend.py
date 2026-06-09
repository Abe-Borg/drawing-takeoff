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
    args = ap.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("ANTHROPIC_API_KEY is not set — M3 is the LLM step and needs a key.", file=sys.stderr)
        return 2

    from . import measure
    from .geometry import extract_pdf_geometry, render_style_swatch

    geom = extract_pdf_geometry(args.pdf, pages=[args.page])[0]
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
