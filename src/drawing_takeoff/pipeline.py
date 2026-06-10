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
from .core import api_config
from .models import (
    SheetGeometry,
    SystemSizeResult,
    SystemSizeSheet,
    TakeoffItem,
    TakeoffResult,
)

__all__ = [
    "extract_takeoff",
    "extract_system_size_takeoff",
    "system_size_for_sheet",
    "write_system_size_export",
]

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


# ---------------------------------------------------------------------------
# System × Size takeoff (M5–M8): pipe networks -> system -> size, aggregated
# across a set. The richer deliverable the legend CLI's --system-size exposes;
# lifted here so the GUI and the CLI run one orchestration (and it stays
# unit-testable with a fake client, like extract_takeoff above).
# ---------------------------------------------------------------------------
def _style_review_notes(
    runs_by, style_labels, all_nets, *, top: int, max_styles: int, ppf: float
) -> list[str]:
    """Surface what the System×Size pass did NOT count, so footage is never
    silently dropped: networks beyond the top-N cap, styles the model called
    maybe-pipe (flagged, not counted), confident non-pipe exclusions, and styles
    ranked below the style cap (never shown to the model)."""
    extra: list[str] = []
    remainder = all_nets[top:]
    if remainder:
        extra.append(
            f"NOT LABELED: {len(remainder)} smaller networks beyond the top {top} = "
            f"{sum((nw.length_ft or 0.0) for nw in remainder):,.0f} LF "
            "(raise the network cap to include them)."
        )
    excluded_n = excluded_lf = unreviewed_n = unreviewed_lf = 0
    for style, rs in runs_by.items():
        if not rs:
            continue
        lab = style_labels.get(style)
        lf = sum(r.length_pt for r in rs) / ppf
        if lab is None:  # ranked below the style cap -> never judged; surface, don't hide
            unreviewed_n += 1
            unreviewed_lf += lf
        elif lab.trusted:
            continue  # confidently pipe -> counted
        elif lab.measurable:  # maybe pipe -> surfaced for confirmation, NOT auto-counted
            extra.append(
                f"STYLE TO CONFIRM (maybe pipe — NOT counted): {lab.system} "
                f"({lf:,.0f} LF) — {lab.reasoning or 'unsure if pipe'}"
            )
        else:
            excluded_n += 1
            excluded_lf += lf
    if excluded_n:
        extra.append(f"EXCLUDED as non-pipe by the style pass: {excluded_n} style(s) = {excluded_lf:,.0f} LF.")
    if unreviewed_n:
        extra.append(
            f"NOT REVIEWED (ranked below the style cap of {max_styles}): "
            f"{unreviewed_n} style(s) = {unreviewed_lf:,.0f} LF — raise the style cap to classify them."
        )
    return extra


def system_size_for_sheet(
    geom: SheetGeometry,
    *,
    client=None,
    scale_label: str | None = None,
    discipline: str = "construction",
    max_styles: int = 12,
    top: int = 8,
    second_look: bool = True,
    legend_pdf: bytes | None = None,
    legend_image: bytes | None = None,
    log: Callable[[str], None] | None = None,
) -> SystemSizeSheet:
    """One sheet, end to end into a System×Size table.

    style→pipe (M3) → connected networks (M5) → size callouts (M6) →
    network→system labels with a high-DPI second-look re-check (M7). Returns plain
    data plus the labeled networks (so a marked-up PDF can be re-rendered), so the
    GUI and the ``legend --system-size`` CLI share one orchestration. The LLM only
    ever labels — every linear foot traces back to exact geometry. Rendering reads
    the sheet from ``geom.ref.source``/``page_index``. Raises ``ValueError`` when
    the sheet carries no usable scale (the caller records it as a per-sheet error).
    """
    from .geometry import (  # deferred: the only PyMuPDF touch (renders the marks)
        render_network_crop_png,
        render_networks_png,
        render_style_swatch,
    )

    log = log or _noop
    sheet_id = f"{geom.ref.source}#p{geom.ref.page_index}"
    pdf_path, page_index = geom.ref.source, geom.ref.page_index
    ppf = _resolve_ppf(geom, scale_label)
    if not ppf:
        raise ValueError(f"{sheet_id}: no scale detected (confirm the scale and re-run)")
    if client is None:
        from .client import get_client

        client = get_client()

    runs_by = measure.runs_by_style(geom, ppf=ppf, exclude_border=True)

    # Pass 1 (M3): which styles are pipe — so mains and branches drawn in
    # different pens all feed the takeoff, not just the single heaviest-dark pen.
    cands = legend._candidates(geom, ppf=ppf, max_styles=max_styles)
    swatches = {c.style_key: render_style_swatch(c.style_key) for c in cands}
    log(f"{sheet_id}: scale {ppf:g} pt/ft; labeling {len(cands)} style(s) with Claude… (pipe vs background)")
    style_labels = legend.label_styles(
        geom, client=client, ppf=ppf, discipline=discipline, style_images=swatches,
        legend_pdf=legend_pdf, legend_image=legend_image,
        max_styles=max_styles, model=api_config.MODEL_SONNET_46,
    )
    pipe = legend.pipe_runs_from_style_labels(runs_by, style_labels)
    if not pipe:
        log(f"{sheet_id}: no pipe styles identified on this sheet.")
        return SystemSizeSheet(
            sheet=sheet_id, source=pdf_path, page_index=page_index, networks=[],
            tables={"by_system_size": {}, "detail": [], "review": []},
            report=f"=== System×Size: {sheet_id} ===\n  no pipe styles identified on this sheet.",
        )

    # Pass 2 (M5 + M7): connect the pipe runs into networks and name each system.
    all_nets = measure.networks(pipe, ppf=ppf)
    nets = all_nets[:top]
    image = render_networks_png(pdf_path, page_index, nets)
    log(f"{sheet_id}: {len(nets)} pipe network(s); labeling systems with Claude…")
    # ppf is the override-aware scale (a GUI/CLI manual scale wins over the sheet's
    # own detected one); thread it into the labeling facts so the size callouts the
    # model reads are snapped at the confirmed scale, not the sheet's default.
    labels = legend.label_networks(geom, nets, client=client, image=image, ppf=ppf, discipline=discipline)

    # Second look: re-check only the flagged networks, from high-DPI close-ups of
    # just those regions (the engine knows where each ambiguity lives).
    flagged = [nw for nw in nets if legend.needs_second_look(labels[nw.id])]
    if flagged and second_look:
        log(f"{sheet_id}: second look on {len(flagged)} flagged network(s) with Claude…")
        crops = {nw.id: render_network_crop_png(pdf_path, page_index, nw) for nw in flagged}
        labels.update(
            legend.second_look_networks(geom, flagged, labels, crops, client=client, ppf=ppf, discipline=discipline)
        )

    notes = _style_review_notes(runs_by, style_labels, all_nets, top=top, max_styles=max_styles, ppf=ppf)
    report = legend.build_system_size_report(nets, labels, geom, ppf=ppf)
    if notes:
        report += "\n\n" + "\n".join(notes)
    tables = legend.takeoff_tables(nets, labels, geom, ppf=ppf)
    tables["review"] = list(tables["review"]) + notes
    return SystemSizeSheet(
        sheet=sheet_id, source=pdf_path, page_index=page_index, networks=nets,
        tables=tables, report=report, notes=notes,
    )


def _absorb_sheet(result: SystemSizeResult, sheet: SystemSizeSheet) -> None:
    """Fold one sheet into the set-wide totals: sum the System×Size table, tag each
    detail row with its sheet, and prefix review notes with the sheet id."""
    result.sheets.append(sheet)
    for key, lf in sheet.tables.get("by_system_size", {}).items():
        result.by_system_size[key] = result.by_system_size.get(key, 0.0) + lf
    for row in sheet.tables.get("detail", []):
        result.detail.append({"sheet": sheet.sheet, **row})
    for note in sheet.tables.get("review", []):
        result.review.append(f"{sheet.sheet}: {note}")


def extract_system_size_takeoff(
    pdf_paths: Sequence[str | Path],
    *,
    client=None,
    progress: Callable[[int, int, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    scale_label: str | None = None,
    discipline: str = "construction",
    max_styles: int = 12,
    top: int = 8,
    second_look: bool = True,
    legend_pdf: bytes | None = None,
    legend_image: bytes | None = None,
) -> SystemSizeResult:
    """Run a System×Size takeoff over ``pdf_paths`` and aggregate across the set.

    Each sheet is taken end to end by :func:`system_size_for_sheet` (style→pipe →
    networks → size → system, with a second-look re-check); the per-sheet
    System×Size tables are summed into one set-wide total, and every sheet's
    labeled networks are retained so :func:`write_system_size_export` can emit a
    marked-up PDF per sheet. One unreadable sheet is recorded in ``errors`` and
    skipped — it never sinks the run.

    Args mirror :func:`extract_takeoff`, plus the System×Size knobs: ``top``
    (largest N networks labeled per sheet), ``max_styles`` (styles classified per
    sheet), and ``second_look`` (the high-DPI re-check of flagged networks).
    """
    from .geometry import extract_pdf_geometry  # deferred: only this needs PyMuPDF

    log = log or _noop
    result = SystemSizeResult()
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
    if client is None and total:
        from .client import get_client  # resolve once, share across sheets

        client = get_client()

    for i, geom in enumerate(sheets):
        name = f"{Path(geom.ref.source).name} p{geom.ref.page_index}"
        if progress is not None:
            progress(i, total, name)
        log(f"[{i + 1}/{total}] {name}")
        try:
            sheet = system_size_for_sheet(
                geom, client=client, scale_label=scale_label, discipline=discipline,
                max_styles=max_styles, top=top, second_look=second_look,
                legend_pdf=legend_pdf, legend_image=legend_image, log=log,
            )
            _absorb_sheet(result, sheet)
            result.diagnostics.append(f"{sheet.sheet}: {len(sheet.networks)} network(s) labeled")
        except Exception as exc:
            result.errors.append(f"{geom.ref.source}#p{geom.ref.page_index}: {exc}")
            result.diagnostics.append(f"{geom.ref.source}#p{geom.ref.page_index}: ERROR {exc}")
            log(f"{name}: ERROR {exc}")

    if progress is not None:
        progress(total, total, "done")
    return result


def write_system_size_export(
    result: SystemSizeResult, out_dir: str | Path = ".", *, project_name: str = "takeoff"
) -> Path:
    """Write the System×Size deliverables into a ``<slug>_<timestamp>`` folder.

    One Excel workbook (``takeoff.xlsx`` — Summary / Detail / Review, aggregated
    across the set) plus one marked-up PDF per sheet (``<stem>_p<page>_markup.pdf``,
    colored + numbered networks on the original page). Returns the folder.
    """
    from datetime import datetime

    from . import export
    from .geometry import write_marked_up_pdf  # deferred: PyMuPDF for the markup

    folder = Path(out_dir) / f"{export._slug(project_name)}_{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    export.build_takeoff_workbook(result.as_tables()).save(folder / "takeoff.xlsx")

    used: set[str] = set()
    for sheet in result.sheets:
        if not sheet.networks:  # nothing to mark up on a sheet with no pipe
            continue
        name = f"{Path(sheet.source).stem}_p{sheet.page_index}_markup.pdf"
        n = 2
        while name in used:  # two inputs can share a stem+page across folders
            name = f"{Path(sheet.source).stem}_p{sheet.page_index}_{n}_markup.pdf"
            n += 1
        used.add(name)
        write_marked_up_pdf(sheet.source, sheet.page_index, sheet.networks, str(folder / name))
    return folder
