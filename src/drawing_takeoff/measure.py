"""Linear measurement primitives — pure geometry math, no PyMuPDF, no LLM.

Turns the backend-free :class:`~drawing_takeoff.models.SheetGeometry` into
per-style linear footage you can trust:

  * length of a polyline / cubic Bezier in points,
  * grouping paths by exact :class:`~drawing_takeoff.models.StyleKey`,
  * **stitching** the fragments a CAD export splits one straight run into back
    into a single :class:`~drawing_takeoff.models.Run` (M1 showed the naive
    "nearest fragment" under-measures and naive "connect everything" overshoots
    through tees — so the stitcher unions only fragments that are *connected
    AND collinear*),
  * per-style total feet, and the text<->run association that bridges a label
    to the run it annotates.

Everything operates on plain ``(x, y)`` float tuples, so it is fully
unit-testable without a PDF backend.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable, Sequence

from . import scale as _scale
from .models import BBox, GeometryPath, Point, Run, SheetGeometry, StyleKey

_BEZIER_SAMPLES = 16
_DEFAULT_TOL = 0.5          # pt: endpoints within this snap to one node
_DEFAULT_ANGLE_TOL = 2.0    # deg: segments within this of each other are collinear


# ---------------------------------------------------------------------------
# length primitives
# ---------------------------------------------------------------------------
def segment_length_pt(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def polyline_length_pt(points: Sequence[Point]) -> float:
    """Sum of Euclidean distances between consecutive points."""
    return sum(segment_length_pt(points[i], points[i + 1]) for i in range(len(points) - 1))


def _sample_bezier(p0: Point, p1: Point, p2: Point, p3: Point, samples: int) -> list[Point]:
    pts = [p0]
    for k in range(1, samples + 1):
        t = k / samples
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt*mt*t * p1[0] + 3 * mt*t*t * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt*mt*t * p1[1] + 3 * mt*t*t * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def bezier_length_pt(
    p0: Point, p1: Point, p2: Point, p3: Point, samples: int = _BEZIER_SAMPLES
) -> float:
    """Cubic Bezier arc length, approximated by ``samples`` chords.

    PyMuPDF gives no arc length, so we sample the cubic and sum chord lengths;
    N≈16 is plenty for pipe bends. Most runs are straight — curves are the
    exception (the sample sheets are 100% straight segments).
    """
    return polyline_length_pt(_sample_bezier(p0, p1, p2, p3, samples))


# ---------------------------------------------------------------------------
# style grouping
# ---------------------------------------------------------------------------
def group_by_style(paths: Iterable[GeometryPath]) -> dict[StyleKey, list[GeometryPath]]:
    """Group paths by exact style — the propagation key for labeling."""
    out: dict[StyleKey, list[GeometryPath]] = defaultdict(list)
    for p in paths:
        out[p.style_key].append(p)
    return dict(out)


# ---------------------------------------------------------------------------
# stitching
# ---------------------------------------------------------------------------
def _straight_segments(paths: Iterable[GeometryPath], *, bezier_samples: int) -> list[tuple[Point, Point]]:
    """Every straight segment of ``paths`` ('l' as-is, 'c' flattened to chords)."""
    segs: list[tuple[Point, Point]] = []
    for p in paths:
        for it in p.items:
            if it[0] == "l":
                segs.append((it[1], it[2]))
            elif it[0] == "c":
                pts = _sample_bezier(it[1], it[2], it[3], it[4], bezier_samples)
                segs.extend((pts[i], pts[i + 1]) for i in range(len(pts) - 1))
            # 're' / 'qu' are closed shapes, not linear runs — skip.
    return segs


class _NodeIndex:
    """Assigns endpoints to merged nodes, snapping points within ``tol`` together.

    A plain grid-round splits two points that lie within ``tol`` of each other
    but straddle a cell boundary (e.g. ``0.24`` and ``0.26`` round to ``0.0`` and
    ``0.5`` at ``tol=0.5``). This indexes points in ``tol``-sized cells and
    searches the 3x3 neighborhood, so any two points within ``tol`` share a node
    regardless of where the boundary falls — honoring the documented tolerance.
    """

    def __init__(self, tol: float) -> None:
        self.tol = tol
        self._cells: dict[tuple[int, int], list[tuple[Point, int]]] = defaultdict(list)
        self._count = 0

    def node_of(self, p: Point) -> int:
        cx, cy = math.floor(p[0] / self.tol), math.floor(p[1] / self.tol)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q, nid in self._cells.get((cx + dx, cy + dy), ()):
                    if math.hypot(q[0] - p[0], q[1] - p[1]) <= self.tol:
                        return nid
        nid = self._count
        self._count += 1
        self._cells[(cx, cy)].append((p, nid))
        return nid


def _angle_mod180(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


def _angles_collinear(a1: float, a2: float, tol: float) -> bool:
    d = abs(a1 - a2) % 180.0
    return d <= tol or d >= 180.0 - tol


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def stitch_runs(
    paths: Iterable[GeometryPath],
    *,
    tol: float = _DEFAULT_TOL,
    angle_tol_deg: float = _DEFAULT_ANGLE_TOL,
    ppf: float | None = None,
    bezier_samples: int = _BEZIER_SAMPLES,
) -> list[Run]:
    """Stitch a style's fragments into maximally-straight :class:`Run` s.

    Two segments are joined only when they **share an endpoint** (within
    ``tol``) **and are collinear** (within ``angle_tol_deg``). That single
    constraint is what M1's probes proved is needed: connectivity alone runs
    straight through a tee and merges the branch (length explodes); collinearity
    alone can't bridge the fragments. Exact-duplicate segments (double-drawn
    lines) are dropped first so the total isn't inflated.

    ``style_key`` on each returned run is taken from the first contributing
    path's style; callers normally pass one style's paths (see
    :func:`group_by_style`).
    """
    paths = list(paths)
    style_key = paths[0].style_key if paths else None
    raw = _straight_segments(paths, bezier_samples=bezier_samples)

    # Assign each endpoint a node id, merging points within `tol` (honoring the
    # tolerance across grid boundaries — see _NodeIndex). Dedup exact-duplicate
    # segments by their node-id pair so double-drawn lines don't inflate totals.
    index = _NodeIndex(tol)
    seen: set = set()
    segs: list[tuple[Point, Point]] = []
    seg_nodes: list[tuple[int, int]] = []
    for a, b in raw:
        if segment_length_pt(a, b) <= 0:
            continue
        na, nb = index.node_of(a), index.node_of(b)
        if na == nb:
            continue  # endpoints within tol of each other -> a point, not a run
        key = (na, nb) if na <= nb else (nb, na)
        if key in seen:
            continue
        seen.add(key)
        segs.append((a, b))
        seg_nodes.append((na, nb))

    if not segs:
        return []

    angles = [_angle_mod180(a, b) for a, b in segs]
    node_segs: dict[int, list[int]] = defaultdict(list)
    for i, (na, nb) in enumerate(seg_nodes):
        node_segs[na].append(i)
        node_segs[nb].append(i)

    uf = _UnionFind(len(segs))
    for incident in node_segs.values():
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                si, sj = incident[i], incident[j]
                if _angles_collinear(angles[si], angles[sj], angle_tol_deg):
                    uf.union(si, sj)

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(len(segs)):
        comps[uf.find(i)].append(i)

    runs: list[Run] = []
    for members in comps.values():
        length_pt = sum(segment_length_pt(*segs[i]) for i in members)
        pts: list[Point] = []
        for i in members:
            pts.extend(segs[i])
        # order points along the run's dominant direction for a clean polyline
        ang = math.radians(angles[members[0]])
        dx, dy = math.cos(ang), math.sin(ang)
        uniq = sorted(set(pts), key=lambda p: p[0] * dx + p[1] * dy)
        xs = [p[0] for p in uniq]
        ys = [p[1] for p in uniq]
        bbox: BBox = (min(xs), min(ys), max(xs), max(ys))
        runs.append(
            Run(
                style_key=style_key,
                polyline=tuple(uniq),
                length_pt=length_pt,
                bbox=bbox,
                length_ft=(length_pt / ppf) if ppf else None,
                segment_count=len(members),
            )
        )
    return runs


# The sheet border and matchlines span (essentially) the full sheet; a real
# pipe/duct/wall run almost never does — even a 62 ft riser is only ~25% of a
# 42x30 sheet. So "is this the border?" is a GEOMETRIC question — does the run
# span >= this fraction of the page width or height — not a length threshold
# (which is scale-dependent and wrongly drops a long real run). This keeps a
# sheet border out of every total even when it's drawn in the pipe's own pen.
_BORDER_SPAN_FRAC = 0.85


def is_border_run(
    run: Run, page_width_pt: float, page_height_pt: float, *, span_frac: float = _BORDER_SPAN_FRAC
) -> bool:
    """Whether ``run`` is a sheet border / matchline (spans ~the whole sheet)."""
    w = run.bbox[2] - run.bbox[0]
    h = run.bbox[3] - run.bbox[1]
    return w >= span_frac * page_width_pt or h >= span_frac * page_height_pt


def runs_by_style(
    geometry: SheetGeometry,
    *,
    ppf: float | None = None,
    exclude_border: bool = False,
    span_frac: float = _BORDER_SPAN_FRAC,
    **kwargs,
) -> dict[StyleKey, list[Run]]:
    """Stitch every style's paths into runs. ``ppf`` defaults to the sheet's.

    With ``exclude_border``, runs that span ~the whole sheet (the border or a
    matchline, per :func:`is_border_run`) are dropped so a sheet border never
    poses as a measurable system — even one drawn in the pipe's own pen.
    """
    if ppf is None:
        ppf = geometry.points_per_foot
    grouped = group_by_style(geometry.non_degenerate_paths())
    out: dict[StyleKey, list[Run]] = {}
    for k, paths in grouped.items():
        runs = stitch_runs(paths, ppf=ppf, **kwargs)
        if exclude_border:
            runs = [
                r
                for r in runs
                if not is_border_run(r, geometry.page_width_pt, geometry.page_height_pt, span_frac=span_frac)
            ]
        out[k] = runs
    return out


def linear_feet_by_style(
    geometry: SheetGeometry,
    ppf: float | None = None,
    *,
    exclude_border: bool = True,
    span_frac: float = _BORDER_SPAN_FRAC,
) -> dict[StyleKey, float]:
    """Total linear feet per style (sum of stitched-run lengths / ppf).

    ``exclude_border`` defaults to **True**: a takeoff total must never include
    the sheet border or a matchline. Pass ``False`` for the raw geometry total.
    """
    if ppf is None:
        ppf = geometry.points_per_foot
    if not ppf:
        raise ValueError("points_per_foot is required (sheet has no scale)")
    out: dict[StyleKey, float] = {}
    runs_by = runs_by_style(geometry, ppf=ppf, exclude_border=exclude_border, span_frac=span_frac)
    for style, runs in runs_by.items():
        total = sum(r.length_pt for r in runs) / ppf
        if total > 0:
            out[style] = total
    return out


# ---------------------------------------------------------------------------
# text <-> run association (the bridge from a label to the run it tags)
# ---------------------------------------------------------------------------
def nearest_run(point: Point, runs: Sequence[Run], *, max_dist_pt: float | None = None) -> Run | None:
    """The run whose bbox-centroid is closest to ``point`` (within ``max_dist_pt``)."""
    best: Run | None = None
    best_d = math.inf
    for r in runs:
        cx, cy = r.centroid
        d = math.hypot(cx - point[0], cy - point[1])
        if d < best_d:
            best_d, best = d, r
    if best is not None and max_dist_pt is not None and best_d > max_dist_pt:
        return None
    return best


# ---------------------------------------------------------------------------
# M2 report: per-style footage + validation against length tags
# ---------------------------------------------------------------------------
def _fmt_style(k: StyleKey) -> str:
    col = "none" if k.stroke_color is None else ",".join(f"{c:.2f}" for c in k.stroke_color)
    return f"[{col}] w={k.width} {k.dashes}"


def _length_tags(geometry: SheetGeometry):
    """``(text, feet, centroid)`` for every NN-NN pipe-length tag on the sheet."""
    out = []
    for w in geometry.words:
        if re.fullmatch(r"\d+-\d+", w.text):
            ft = _scale.parse_feet_inches(w.text, allow_bare_hyphen=True)
            if ft is not None and 1.0 <= ft <= 60.0:
                out.append((w.text, ft, w.centroid))
    return out


def length_tag_total(geometry: SheetGeometry) -> tuple[int, float]:
    """``(count, total_feet)`` of the drawing's NN-NN pipe-length callouts.

    The engineer's own stated total — an independent ground truth to cross-check
    the measured total against. (Associating *which* tag belongs to *which* run
    is the harder text<->run problem, deferred to M3; the takeoff only needs the
    sum.)
    """
    n, total = 0, 0.0
    for _text, feet, _centroid in _length_tags(geometry):
        n += 1
        total += feet
    return n, total


def heaviest_dark_style(styles: Iterable[StyleKey]) -> StyleKey | None:
    """The thickest near-black stroke style — a heuristic for the takeoff
    lineweight when there's no legend yet (the legend/user confirms in M3/M4)."""
    dark = [s for s in styles if s.stroke_color is not None and max(s.stroke_color) < 0.30 and s.width > 0]
    return max(dark, key=lambda s: s.width) if dark else None


def build_measure_report(
    geometry: SheetGeometry, *, ppf: float | None = None, span_frac: float = _BORDER_SPAN_FRAC
) -> str:
    if ppf is None:
        ppf = geometry.points_per_foot
    lines = [f"=== M2 linear measurement: {geometry.ref.source} (page {geometry.ref.page_index}) ==="]
    if not ppf:
        return "\n".join(lines + ["  no scale on sheet; cannot compute footage"])
    lines.append(f"scale: {geometry.scale_label!r} -> {ppf:.4g} pt/ft")

    runs_by = runs_by_style(geometry, ppf=ppf)  # raw; border filtered per-style below
    pw, ph = geometry.page_width_pt, geometry.page_height_pt

    def _split(runs):
        kept = [r for r in runs if not is_border_run(r, pw, ph, span_frac=span_frac)]
        dropped = [r for r in runs if is_border_run(r, pw, ph, span_frac=span_frac)]
        return kept, dropped

    rows = []
    for style, runs in runs_by.items():
        kept, _dropped = _split(runs)
        ft = sum(r.length_pt for r in kept) / ppf
        if ft > 0:
            rows.append((style, len(kept), ft))
    rows.sort(key=lambda r: -r[2])

    lines.append("")
    lines.append(f"  {'per-style linear footage, border excluded  [stroke_rgb] width dashes':62s} {'runs':>6s} {'total_ft':>11s}")
    for style, nruns, total_ft in rows[:12]:
        lines.append(f"    {_fmt_style(style):60s} {nruns:6d} {total_ft:11,.1f}")

    # Highlight the likely takeoff lineweight, picked INDEPENDENTLY of the tags
    # (heaviest dark pen) so the cross-check below isn't circular.
    pipe_style = heaviest_dark_style(runs_by.keys())
    if pipe_style is not None:
        pipe_runs, border_runs = _split(runs_by[pipe_style])
        pipe_ft = sum(r.length_pt for r in pipe_runs) / ppf
        longest = max((r.length_pt / ppf for r in pipe_runs), default=0.0)
        lines += [
            "",
            f"  TAKEOFF lineweight (heaviest dark pen): {_fmt_style(pipe_style)}",
            f"    {len(pipe_runs)} runs, {pipe_ft:,.1f} LF  "
            f"(dropped {len(border_runs)} full-sheet-spanning run(s) as border/matchline; "
            f"longest kept run {longest:.1f} ft)",
        ]
        n_tags, tag_ft = length_tag_total(geometry)
        if n_tags:
            pct = 100.0 * (pipe_ft - tag_ft) / tag_ft
            lines.append(
                f"  CROSS-CHECK vs the drawing's own callouts: sum of {n_tags} length tags "
                f"= {tag_ft:,.1f} LF  ->  measured total agrees to {pct:+.2f}%"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="M2 linear measurement report for a vector sheet.")
    ap.add_argument("pdf", help="path to the vector PDF sheet")
    ap.add_argument("--page", type=int, default=0, help="0-based page index (default 0)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args(argv)

    from .geometry import extract_pdf_geometry  # deferred: only the CLI needs PyMuPDF

    geom = extract_pdf_geometry(args.pdf, pages=[args.page])[0]
    report = build_measure_report(geom)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
