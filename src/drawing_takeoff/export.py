"""Takeoff output: pure document builders + a thin file-writing wrapper.

``build_takeoff_documents`` is pure — ``TakeoffResult`` in, a list of
``(filename, text)`` out — so the CSV/diagnostics shape is unit-testable without
touching the filesystem. ``write_takeoff_export`` is the only I/O: it drops those
documents into a timestamped folder. Mirrors the sibling project's
``build_*_documents`` / ``write_*`` split.

Three documents:
  * ``takeoff_by_system.csv`` — the headline: trusted LF totaled by system.
  * ``takeoff_detail.csv``    — one row per (sheet, style), with provenance and
                                a FLAGGED column for ambiguous styles.
  * ``diagnostics.txt``       — scales, per-sheet notes, errors, flagged styles.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from pathlib import Path

from .models import StyleKey, TakeoffResult


def _fmt_style(k: StyleKey) -> str:
    col = "none" if k.stroke_color is None else ",".join(f"{c:.2f}" for c in k.stroke_color)
    return f"[{col}] w={k.width} {k.dashes}"


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")
    return s or "takeoff"


def _by_system_csv(result: TakeoffResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["System", "Quantity", "Unit", "Sheets", "Styles", "Min_Confidence"])
    by: dict[str, dict] = {}
    for it in result.trusted_items:
        d = by.setdefault(it.system, {"qty": 0.0, "sheets": set(), "styles": set(), "conf": set()})
        d["qty"] += it.quantity
        d["sheets"].add(it.sheet)
        d["styles"].add(it.style_key)
        d["conf"].add(it.confidence)
    order = {"high": 0, "medium": 1, "low": 2}
    for system, d in sorted(by.items(), key=lambda kv: -kv[1]["qty"]):
        worst = max(d["conf"], key=lambda c: order.get(c, 3)) if d["conf"] else ""
        w.writerow([system, f"{d['qty']:.1f}", "LF", len(d["sheets"]), len(d["styles"]), worst])
    return buf.getvalue()


def _detail_csv(result: TakeoffResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        ["System", "Sheet", "Lineweight", "Quantity", "Unit", "Confidence",
         "Flagged", "Runs", "Scale_pt_per_ft", "Notes"]
    )
    for it in sorted(result.items, key=lambda i: (i.sheet, -i.quantity)):
        w.writerow([
            it.system, it.sheet, _fmt_style(it.style_key), f"{it.quantity:.1f}", it.unit,
            it.confidence, "YES" if not it.trusted else "", it.run_count,
            f"{it.scale_used:g}", (it.reasoning or "").replace("\n", " "),
        ])
    return buf.getvalue()


def _diagnostics_txt(result: TakeoffResult) -> str:
    lines = [
        "=== drawing-takeoff diagnostics ===",
        f"sheets processed: {result.sheet_count}",
        f"takeoff items: {len(result.items)}  (trusted: {len(result.trusted_items)}, "
        f"flagged: {len(result.flagged)})",
        "",
        "TOTALS by system (trusted):",
    ]
    if result.per_system_totals:
        lines += [f"  {sys}: {qty:,.1f} LF" for sys, qty in result.per_system_totals.items()]
    else:
        lines.append("  (none)")
    if result.flagged:
        lines += ["", "FLAGGED for review (not counted):"]
        for it in result.flagged:
            lines.append(
                f"  {it.sheet}  {_fmt_style(it.style_key)}  {it.quantity:,.1f} LF  "
                f"-> {it.system} [conf={it.confidence}, ambiguous={it.ambiguous}]"
                + (f"  — {it.reasoning}" if it.reasoning else "")
            )
    if result.errors:
        lines += ["", "ERRORS:"] + [f"  {e}" for e in result.errors]
    lines += ["", "per-sheet trail:"] + [f"  {d}" for d in result.diagnostics]
    return "\n".join(lines) + "\n"


def build_takeoff_documents(result: TakeoffResult) -> list[tuple[str, str]]:
    """Pure: a takeoff result -> ``[(filename, text), ...]`` (no I/O)."""
    return [
        ("takeoff_by_system.csv", _by_system_csv(result)),
        ("takeoff_detail.csv", _detail_csv(result)),
        ("diagnostics.txt", _diagnostics_txt(result)),
    ]


def write_takeoff_export(
    result: TakeoffResult, out_dir: str | Path = ".", *, project_name: str = "takeoff"
) -> Path:
    """Write the takeoff documents into a ``<slug>_<timestamp>`` folder; return it."""
    folder = Path(out_dir) / f"{_slug(project_name)}_{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in build_takeoff_documents(result):
        (folder / name).write_text(content, encoding="utf-8")
    return folder
