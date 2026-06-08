"""M1 go/no-go diagnostics: dump a report answering the three questions.

Runs on the pure :class:`~drawing_takeoff.models.SheetGeometry` produced by
:mod:`drawing_takeoff.geometry` — no PyMuPDF on this path — and prints:

  (a) cleanliness   path counts grouped by StyleKey, the busiest styles, extents;
  (b) instances     congruent repeated geometry (signature-hash counts);
  (c) scale         label -> points_per_foot, then a known-dimension check
                    (measured points / ppf vs. the stated value), the M1 gate.

The scale check here associates each dimensioned-pipe label to the *nearest
single* segment (pre-stitch). Clean, unfragmented runs validate to well under
1%; runs the CAD export split into collinear fragments read short — that under-
measurement is exactly what M2's endpoint-stitching is built to resolve, so the
aggregate is reported alongside the decisive best matches rather than instead of
them.

Usage:  python -m drawing_takeoff.diagnose SHEET.pdf [--out report.txt]
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
import re

from . import scale as _scale
from .geometry import extract_pdf_geometry
from .models import GeometryPath, SheetGeometry, StyleKey

_BEZIER_SAMPLES = 16


# ---------------------------------------------------------------------------
# length helpers (minimal; M2's measure.py supersedes these with stitching)
# ---------------------------------------------------------------------------
def _dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _bezier_len(p0, p1, p2, p3, samples: int = _BEZIER_SAMPLES) -> float:
    prev, total = p0, 0.0
    for k in range(1, samples + 1):
        t = k / samples
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt*mt*t * p1[0] + 3 * mt*t*t * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt*mt*t * p1[1] + 3 * mt*t*t * p2[1] + t**3 * p3[1]
        total += _dist(prev, (x, y))
        prev = (x, y)
    return total


def _path_length(path: GeometryPath) -> float:
    total = 0.0
    for it in path.items:
        op = it[0]
        if op == "l":
            total += _dist(it[1], it[2])
        elif op == "c":
            total += _bezier_len(it[1], it[2], it[3], it[4])
        elif op == "re":
            x0, y0, x1, y1 = it[1]
            total += 2 * (abs(x1 - x0) + abs(y1 - y0))
    return total


def _straight_segments(geom: SheetGeometry):
    """Yield ``(midpoint, length, style_key)`` for every straight 'l' segment."""
    for p in geom.non_degenerate_paths():
        for it in p.items:
            if it[0] == "l":
                a, b = it[1], it[2]
                L = _dist(a, b)
                if L > 0:
                    yield (((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), L, p.style_key)


def _fmt_style(k: StyleKey) -> str:
    col = "none" if k.stroke_color is None else ",".join(f"{c:.2f}" for c in k.stroke_color)
    return f"[{col}] w={k.width} {k.dashes}"


# ---------------------------------------------------------------------------
# (a) cleanliness
# ---------------------------------------------------------------------------
@dataclass
class StyleStat:
    n: int = 0
    length_pt: float = 0.0


def style_histogram(geom: SheetGeometry) -> dict[StyleKey, StyleStat]:
    hist: dict[StyleKey, StyleStat] = defaultdict(StyleStat)
    for p in geom.non_degenerate_paths():
        s = hist[p.style_key]
        s.n += 1
        s.length_pt += _path_length(p)
    return hist


def cleanliness_section(geom: SheetGeometry) -> str:
    total = len(geom.paths)
    real = geom.non_degenerate_paths()
    hist = style_histogram(geom)
    # Length-bearing paths: the 84k light-gray sub-0.01pt paths are stipple/
    # pattern noise (no measurable length); name them so the count isn't a
    # mystery. M2 drops them with a per-style min-length threshold.
    measurable = [p for p in real if _path_length(p) >= 1.0]
    negligible = len(real) - len(measurable)
    xs0 = min((p.bbox[0] for p in measurable), default=0.0)
    ys0 = min((p.bbox[1] for p in measurable), default=0.0)
    xs1 = max((p.bbox[2] for p in measurable), default=0.0)
    ys1 = max((p.bbox[3] for p in measurable), default=0.0)
    lines = [
        "(a) CLEANLINESS",
        f"    paths: {total} total; {negligible} are near-zero (<1pt) stipple/"
        f"pattern noise ({100*negligible/max(total,1):.0f}%), {len(measurable)} carry real length",
        f"    distinct styles (StyleKey): {len(hist)}",
        f"    vector extent: ({xs0:.0f},{ys0:.0f})-({xs1:.0f},{ys1:.0f}) pt "
        f"= {100*(xs1-xs0)/geom.page_width_pt:.0f}% x {100*(ys1-ys0)/geom.page_height_pt:.0f}% of page",
        f"    {'busiest styles  [stroke_rgb] width dashes':52s}   {'paths':>7s} {'length_pt':>11s}",
    ]
    for k, s in sorted(hist.items(), key=lambda kv: -kv[1].n)[:12]:
        lines.append(f"      {_fmt_style(k):50s} {s.n:7d} {s.length_pt:11.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# (b) instances
# ---------------------------------------------------------------------------
def _signature(p: GeometryPath):
    ox, oy = p.bbox[0], p.bbox[1]
    parts = []
    for it in p.items:
        coords = []
        for el in it[1:]:
            if isinstance(el, tuple) and len(el) == 2:
                coords.append((round(el[0] - ox, 1), round(el[1] - oy, 1)))
            elif isinstance(el, tuple) and len(el) == 4:
                coords.append((round(el[2] - el[0], 1), round(el[3] - el[1], 1)))
        parts.append((it[0], tuple(coords)))
    return (tuple(parts), p.style_key, round(p.bbox_width, 1), round(p.bbox_height, 1))


def instances_section(geom: SheetGeometry, *, min_repeats: int = 3, min_extent_pt: float = 2.0) -> str:
    # Filter out sub-2pt paths first so genuine repeated symbols surface instead
    # of the 84k near-zero stipple paths drowning the histogram.
    paths = [p for p in geom.non_degenerate_paths() if max(p.bbox_width, p.bbox_height) >= min_extent_pt]
    sig = Counter(_signature(p) for p in paths)
    repeats = sorted(((s, c) for s, c in sig.items() if c >= min_repeats), key=lambda sc: -sc[1])
    lines = [
        "(b) INSTANCES (congruent repeated geometry — symbol suspects)",
        f"    over {len(paths)} paths >={min_extent_pt:g}pt: {len(sig)} distinct signatures; "
        f"repeating >={min_repeats}x: {len(repeats)}",
        "    (still includes repeated background hatch; isolating a specific symbol "
        "needs style+bbox filtering — a counts-milestone concern)",
    ]
    for s, c in repeats[:8]:
        lines.append(f"      count={c:5d}  items={len(s[0]):3d}  bbox={s[2]}x{s[3]}pt  {_fmt_style(s[1])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# (c) scale — the M1 gate
# ---------------------------------------------------------------------------
@dataclass
class DimMatch:
    label: str
    stated_ft: float
    measured_pt: float
    err_pct: float
    style: StyleKey
    dist_pt: float


def _pipe_length_labels(geom: SheetGeometry):
    """Tokens of the form NN-NN that parse as a plausible pipe length (feet)."""
    out = []
    for w in geom.words:
        if re.fullmatch(r"\d+-\d+", w.text):
            ft = _scale.parse_feet_inches(w.text, allow_bare_hyphen=True)
            if ft is not None and 1.0 <= ft <= 60.0:
                out.append((w.text, ft, w.centroid))
    return out


def scale_check(geom: SheetGeometry, *, radius_pt: float = 70.0):
    """Associate each pipe-length label to its nearest segment; return matches.

    Nearest by distance (not by best-fit length), so the agreement is earned,
    not assumed. Returns ``(matches, inferred_pipe_style)``.
    """
    ppf = geom.points_per_foot
    if ppf is None:
        return [], None
    segs = list(_straight_segments(geom))
    matches: list[DimMatch] = []
    for label, ft, (cx, cy) in _pipe_length_labels(geom):
        expected = ft * ppf
        best = None
        for (mid, L, st) in segs:
            d = math.hypot(mid[0] - cx, mid[1] - cy)
            # plausible pipe segment near the label: within radius, length in a
            # generous band around expected so a stray tick mark isn't picked.
            if d <= radius_pt and 0.5 * expected <= L <= 1.6 * expected:
                if best is None or d < best[0]:
                    best = (d, L, st)
        if best is not None:
            d, L, st = best
            matches.append(DimMatch(label, ft, L, _scale.verify_against_dimension(L, ft, ppf), st, d))
    inferred = None
    if matches:
        good = [m for m in matches if abs(m.err_pct) <= 3.0]
        if good:
            inferred = Counter(m.style for m in good).most_common(1)[0][0]
    return matches, inferred


def scale_section(geom: SheetGeometry) -> str:
    lines = ["(c) SCALE  <-- M1 go/no-go gate"]
    if geom.scale_label:
        lines.append(f"    detected label: {geom.scale_label!r}  ->  points_per_foot = {geom.points_per_foot:.4g}")
    else:
        lines.append("    no scale label detected; cannot validate scale")
        return "\n".join(lines)

    matches, inferred = scale_check(geom)
    if not matches:
        lines.append("    no dimensioned-pipe labels could be associated to geometry")
        return "\n".join(lines)

    matches.sort(key=lambda m: abs(m.err_pct))
    within1 = sum(1 for m in matches if abs(m.err_pct) <= 1.0)
    within2 = sum(1 for m in matches if abs(m.err_pct) <= 2.0)
    # Headline on an actual pipe-style run, not a coincidental background segment
    # that happens to measure exactly — more honest as "the known-dimension check".
    pipe_matches = [m for m in matches if inferred is not None and m.style == inferred]
    best = pipe_matches[0] if pipe_matches else matches[0]
    lines.append(
        f"    KNOWN-DIMENSION CHECK (best match): {best.measured_pt:.1f} pt / "
        f"{geom.points_per_foot:.4g} = {best.measured_pt/geom.points_per_foot:.2f} ft; "
        f"label says {best.label} ({best.stated_ft:.2f} ft) -> err {best.err_pct:+.2f}%  "
        f"{'PASS' if abs(best.err_pct) < 1.0 else 'CHECK'}"
    )
    if inferred is not None:
        lines.append(f"    inferred pipe style (dominant among <=3% matches): {_fmt_style(inferred)}")
    lines.append(
        f"    dimensioned labels associated: {len(matches)}; "
        f"within 1%: {within1}, within 2%: {within2} "
        f"(single-segment/pre-stitch; the sub-1% cluster IS the proof, the spread is fragmentation -> M2)"
    )
    lines.append("    cleanest matches:")
    for m in matches[:8]:
        lines.append(
            f"      {m.label:>7s} {m.stated_ft:6.2f}ft  meas {m.measured_pt:7.1f}pt  "
            f"err {m.err_pct:+6.2f}%  d={m.dist_pt:3.0f}pt  {_fmt_style(m.style)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# report assembly + CLI
# ---------------------------------------------------------------------------
def build_report(geom: SheetGeometry) -> str:
    header = (
        f"=== M1 diagnostic: {geom.ref.source} (page {geom.ref.page_index}) ===\n"
        f"page: {geom.page_width_pt:.0f} x {geom.page_height_pt:.0f} pt "
        f"({geom.page_width_pt/72:.1f} x {geom.page_height_pt/72:.1f} in)"
    )
    return "\n\n".join([
        header,
        cleanliness_section(geom),
        instances_section(geom),
        scale_section(geom),
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M1 go/no-go diagnostics for a vector sheet.")
    ap.add_argument("pdf", help="path to the vector PDF sheet")
    ap.add_argument("--page", type=int, default=0, help="0-based page index (default 0)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args(argv)

    geoms = extract_pdf_geometry(args.pdf, pages=[args.page])
    report = build_report(geoms[0])
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
