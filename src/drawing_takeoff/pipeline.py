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


def _noop(_message: str) -> None:
    """Default ``log`` sink — swallows messages when no callback is supplied."""


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
    legend_block: dict | None = None,
    discipline: str = "construction",
    log: Callable[[str], None] | None = None,
) -> tuple[list[TakeoffItem], list[str]]:
    """Measure + label one sheet into ``(items, diagnostics)``.

    Pure with respect to PyMuPDF — operates on a :class:`SheetGeometry` and a
    duck-typed ``client`` — so it is fully unit-testable. Raises if the sheet has
    no usable scale (the caller records it as a per-sheet error). ``log`` is an
    optional sink for human-readable sub-step messages (e.g. "labeling N styles
    with Claude…") so a front-end can show the run is alive during the long
    vision call.
    """
    log = log or _noop
    sheet_id = f"{geom.ref.source}#p{geom.ref.page_index}"
    ppf = _resolve_ppf(geom, scale_label)
    diagnostics = [
        f"{sheet_id}: scale={geom.scale_label or scale_label or 'unknown'} "
        f"ppf={ppf if ppf else 'n/a'}"
    ]
    if not ppf:
        raise ValueError(f"{sheet_id}: no scale detected (confirm the scale and re-run)")
    log(f"{sheet_id}: scale {geom.scale_label or scale_label or 'unknown'} → {ppf:g} pt/ft; measuring geometry…")

    # M2: border-excluded runs per style -> footage + run counts (one stitch pass).
    runs_by = measure.runs_by_style(geom, ppf=ppf, exclude_border=True)
    feet = {s: sum(r.length_pt for r in rs) / ppf for s, rs in runs_by.items() if rs}
    n_styles = min(max(len(feet), 1), _MAX_LABELLED_STYLES)
    log(
        f"{sheet_id}: {len(feet)} line style(s), "
        f"{sum(len(rs) for rs in runs_by.values())} run(s) measured; "
        f"labeling {n_styles} with Claude… (this can take a moment)"
    )

    # M3: name EVERY measured style (cap high enough that footage is never
    # silently dropped by the label limit). The legend is advisory; anything not
    # *confidently* background is flagged for review, never guessed or discarded.
    labels = legend.label_styles(
        geom, client=client, ppf=ppf, discipline=discipline, legend_image=legend_image,
        legend_pdf=legend_pdf, legend_block=legend_block,
        max_styles=n_styles,
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
    msg = f"{sheet_id}: {n_trusted} trusted, {len(items) - n_trusted} flagged, of {len(feet)} measured styles"
    diagnostics.append(msg)
    log(msg)
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
    log: Callable[[str], None] | None = None,
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
        progress: optional ``progress(done, total, label)`` callback — one tick
            per sheet, for a progress bar.
        log: optional ``log(message)`` sink for human-readable detail *within* a
            sheet (reading the PDF, measuring geometry, the long Claude vision
            call, per-sheet results). ``progress`` alone leaves a multi-second
            vision call looking frozen; ``log`` is what a GUI streams so the run
            visibly stays alive.
        scale_label: override applied to every sheet (scale is consistent across
            a set); when ``None`` each sheet's own detected label is used.
        legend_image: optional PNG of the lead sheet's legend (advisory).
        legend_pdf: optional single-page PDF of the legend (advisory; preferred
            over ``legend_image`` — it carries the page's text layer).
        discipline: e.g. ``"fire protection"`` — guides the legend.
    """
    from .geometry import extract_pdf_geometry  # deferred: only this needs PyMuPDF

    log = log or _noop
    result = TakeoffResult()
    sheets: list[SheetGeometry] = []
    for path in pdf_paths:
        log(f"Reading {Path(path).name}…")
        try:
            sheets.extend(extract_pdf_geometry(str(path)))
        except Exception as exc:  # one unreadable PDF must not sink the run
            result.errors.append(f"{path}: could not read PDF ({exc})")
            log(f"{Path(path).name}: could not read PDF ({exc})")

    total = len(sheets)
    result.sheet_count = total
    log(f"{total} sheet(s) to process from {len(pdf_paths)} file(s).")

    # Multi-sheet sets reuse the legend: upload it once via the Files API and
    # reference the file_id from every per-sheet request instead of re-sending
    # the bytes each time. Transport only — the legend stays advisory either
    # way. Any failure falls back to inline bytes (legend_block stays None).
    legend_block = None
    if (legend_pdf or legend_image) and total > 1:
        if client is None:
            from .client import get_client  # deferred: tests inject a fake

            client = get_client()
        log("Uploading the legend once via the Files API…")
        legend_block = legend.upload_legend(client, legend_pdf=legend_pdf, legend_image=legend_image)
        if legend_block is not None:
            msg = (
                f"legend uploaded once via Files API "
                f"(file_id={legend_block['source']['file_id']}, reused on {total} sheets)"
            )
            result.diagnostics.append(msg)
            log(msg)
        else:
            log("Legend upload unavailable; sending it inline with each sheet.")

    try:
        for i, geom in enumerate(sheets):
            name = f"{Path(geom.ref.source).name} p{geom.ref.page_index}"
            if progress is not None:
                progress(i, total, name)
            log(f"[{i + 1}/{total}] {name}")
            try:
                items, diag = takeoff_for_sheet(
                    geom,
                    client=client,
                    scale_label=scale_label,
                    legend_image=legend_image,
                    legend_pdf=legend_pdf,
                    legend_block=legend_block,
                    discipline=discipline,
                    log=log,
                )
                result.items.extend(items)
                result.diagnostics.extend(diag)
            except Exception as exc:
                result.errors.append(str(exc))
                result.diagnostics.append(f"{geom.ref.source}#p{geom.ref.page_index}: ERROR {exc}")
                log(f"{name}: ERROR {exc}")
    finally:
        legend.delete_uploaded_legend(client, legend_block)

    if progress is not None:
        progress(total, total, "done")
    log("Aggregating cross-sheet totals…")
    result.per_system_totals = _aggregate(result.items)
    return result
