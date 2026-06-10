"""Engine orchestration: PDFs in, a named takeoff out.

``extract_takeoff`` is the single, GUI-agnostic entry point (the seam the M0
stub reserved). It ties the milestones together per sheet:

  geometry (M1) -> linear footage by style (M2, border-excluded) ->
  legend labels style->system (M3) -> per-sheet :class:`TakeoffItem` s ->
  cross-sheet totals by system (M4).

It keeps the GUI/engine boundary clean: a ``progress(done, total, label)``
callback, per-sheet error capture (one bad sheet never sinks the run), and a
duck-typed ``client`` for the legend step (injected in tests). The only PyMuPDF
touch is the deferred ``geometry`` import inside the function — every helper
below operates on the pure models, so the assembly is unit-testable with a fake
client and no backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from . import legend, measure
from . import scale as _scale
from .models import SheetGeometry, TakeoffItem, TakeoffResult

__all__ = ["extract_takeoff"]

# Upper bound on styles sent for labeling per sheet. High enough that all real
# measured styles on a normal sheet are labeled (so footage is never dropped by
# the cap); bounded so a pathological style explosion can't balloon the request.
_MAX_LABELLED_STYLES = 40


def _resolve_ppf(geom: SheetGeometry, scale_label: str | None) -> float | None:
    """Scale override (consistent across a set) wins over the sheet's own label."""
    if scale_label:
        return _scale.points_per_foot_from_label(scale_label)
    return geom.points_per_foot


def takeoff_for_sheet(
    geom: SheetGeometry,
    *,
    client=None,
    scale_label: str | None = None,
    legend_image: bytes | None = None,
    legend_pdf: bytes | None = None,
    discipline: str = "construction",
) -> tuple[list[TakeoffItem], list[str]]:
    """Measure + label one sheet into ``(items, diagnostics)``.

    Pure with respect to PyMuPDF — operates on a :class:`SheetGeometry` and a
    duck-typed ``client`` — so it is fully unit-testable. Raises if the sheet has
    no usable scale (the caller records it as a per-sheet error).
    """
    sheet_id = f"{geom.ref.source}#p{geom.ref.page_index}"
    ppf = _resolve_ppf(geom, scale_label)
    diagnostics = [
        f"{sheet_id}: scale={geom.scale_label or scale_label or 'unknown'} "
        f"ppf={ppf if ppf else 'n/a'}"
    ]
    if not ppf:
        raise ValueError(f"{sheet_id}: no scale detected (confirm the scale and re-run)")

    # M2: border-excluded runs per style -> footage + run counts (one stitch pass).
    runs_by = measure.runs_by_style(geom, ppf=ppf, exclude_border=True)
    feet = {s: sum(r.length_pt for r in rs) / ppf for s, rs in runs_by.items() if rs}

    # M3: name EVERY measured style (cap high enough that footage is never
    # silently dropped by the label limit). The legend is advisory; anything not
    # *confidently* background is flagged for review, never guessed or discarded.
    labels = legend.label_styles(
        geom, client=client, ppf=ppf, discipline=discipline, legend_image=legend_image,
        legend_pdf=legend_pdf, max_styles=min(max(len(feet), 1), _MAX_LABELLED_STYLES),
    )

    def _item(style, lf, *, system, confidence, ambiguous, reasoning):
        return TakeoffItem(
            system=system, quantity=round(lf, 1), unit="LF", sheet=sheet_id,
            style_key=style, scale_used=ppf, confidence=confidence, ambiguous=ambiguous,
            run_count=len(runs_by.get(style, [])), reasoning=reasoning,
        )

    items: list[TakeoffItem] = []
    for style, lf in sorted(feet.items(), key=lambda kv: -kv[1]):
        label = labels.get(style)
        if label is None:
            # measured footage beyond the label cap — surface it, never drop.
            items.append(_item(style, lf, system="(unlabeled style)", confidence="low",
                               ambiguous=True, reasoning="measured but not labeled (beyond legend cap)"))
        elif label.measurable:
            items.append(_item(style, lf, system=label.system, confidence=label.confidence,
                               ambiguous=label.ambiguous, reasoning=label.reasoning))
        elif label.ambiguous:
            # the model is unsure it's a run — flag the footage, don't drop it.
            items.append(_item(style, lf, system=label.system or "(uncertain)",
                               confidence=label.confidence, ambiguous=True, reasoning=label.reasoning))
        # else: confidently non-measurable (background / text / symbols) — correctly excluded.

    n_trusted = sum(1 for i in items if i.trusted)
    diagnostics.append(
        f"{sheet_id}: {n_trusted} trusted, {len(items) - n_trusted} flagged, of {len(feet)} measured styles"
    )
    return items, diagnostics


def _aggregate(items: Sequence[TakeoffItem]) -> dict[str, float]:
    """Cross-sheet totals by system — trusted items only, rounded."""
    totals: dict[str, float] = {}
    for it in items:
        if it.trusted:
            totals[it.system] = totals.get(it.system, 0.0) + it.quantity
    return {k: round(v, 1) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}


def extract_takeoff(
    pdf_paths: Sequence[str | Path],
    *,
    client=None,
    progress: Callable[[int, int, str], None] | None = None,
    scale_label: str | None = None,
    legend_image: bytes | None = None,
    legend_pdf: bytes | None = None,
    discipline: str = "construction",
) -> TakeoffResult:
    """Run a linear-length takeoff over ``pdf_paths`` and return a result.

    Args:
        pdf_paths: input PDF sheet paths, processed in page order.
        client: duck-typed Anthropic client for the legend step (injected in
            tests); ``None`` resolves the vendored default at call time.
        progress: optional ``progress(done, total, label)`` callback.
        scale_label: override applied to every sheet (scale is consistent across
            a set); when ``None`` each sheet's own detected label is used.
        legend_image: optional PNG of the lead sheet's legend (advisory).
        legend_pdf: optional single-page PDF of the legend (advisory; preferred
            over ``legend_image`` — it carries the page's text layer).
        discipline: e.g. ``"fire protection"`` — guides the legend.
    """
    from .geometry import extract_pdf_geometry  # deferred: only this needs PyMuPDF

    result = TakeoffResult()
    sheets: list[SheetGeometry] = []
    for path in pdf_paths:
        try:
            sheets.extend(extract_pdf_geometry(str(path)))
        except Exception as exc:  # one unreadable PDF must not sink the run
            result.errors.append(f"{path}: could not read PDF ({exc})")

    total = len(sheets)
    result.sheet_count = total
    for i, geom in enumerate(sheets):
        if progress is not None:
            progress(i, total, f"{Path(geom.ref.source).name} p{geom.ref.page_index}")
        try:
            items, diag = takeoff_for_sheet(
                geom,
                client=client,
                scale_label=scale_label,
                legend_image=legend_image,
                legend_pdf=legend_pdf,
                discipline=discipline,
            )
            result.items.extend(items)
            result.diagnostics.extend(diag)
        except Exception as exc:
            result.errors.append(str(exc))
            result.diagnostics.append(f"{geom.ref.source}#p{geom.ref.page_index}: ERROR {exc}")

    if progress is not None:
        progress(total, total, "done")
    result.per_system_totals = _aggregate(result.items)
    return result
