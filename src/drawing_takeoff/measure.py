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
from collections import Counter, defaultdict
from typing import Iterable, Sequence

from . import scale as _scale
from .models import BBox, GeometryPath, Network, Point, Run, SheetGeometry, StyleKey

_BEZIER_SAMPLES = 16
_DEFAULT_TOL = 0.5          # pt: endpoints within this snap to one node
_DEFAULT_ANGLE_TOL = 2.0    # deg: segments within this of each other are collinear
_DEFAULT_NETWORK_TOL_FT = 0.5  # ft: scale-aware gap bridge for network connectivity (M5 probe)


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


# The sheet border / perimeter rule does two things a real run almost never
# does together: it spans ~the whole sheet AND hugs a page edge (axis-aligned at
# x~=0/W or y~=0/H). Testing span alone would wrongly drop a legitimate long main
# that merely crosses the plan interior (a horizontal feed from x=100 to x=2700
# on a 3024pt-wide sheet), so the edge requirement is what keeps real mains. The
# test is geometric — not a length threshold (scale-dependent, drops long runs).
_BORDER_SPAN_FRAC = 0.85
_BORDER_EDGE_FRAC = 0.06


def is_border_run(
    run: Run,
    page_width_pt: float,
    page_height_pt: float,
    *,
    span_frac: float = _BORDER_SPAN_FRAC,
    edge_frac: float = _BORDER_EDGE_FRAC,
) -> bool:
    """Whether ``run`` is a sheet border / perimeter rule.

    A full-width, near-horizontal run hugging the top or bottom edge, or a
    full-height, near-vertical run hugging the left or right edge. A long main
    that crosses the plan interior (or a diagonal) spans wide but doesn't hug an
    edge, so it is kept.
    """
    x0, y0, x1, y1 = run.bbox
    w, h = x1 - x0, y1 - y0
    mx, my = edge_frac * page_width_pt, edge_frac * page_height_pt
    if w >= span_frac * page_width_pt and h <= my and (y0 <= my or y1 >= page_height_pt - my):
        return True
    if h >= span_frac * page_height_pt and w <= mx and (x0 <= mx or x1 >= page_width_pt - mx):
        return True
    return False


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
# network connectivity (M5): runs -> connected systems ("follow the line")
# ---------------------------------------------------------------------------
def _point_segment_dist(p: Point, a: Point, b: Point) -> float:
    """Shortest distance from point ``p`` to segment ``ab`` (clamped to the ends)."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    seg2 = abx * abx + aby * aby
    if seg2 <= 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / seg2
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby))


def _run_segments(run: Run) -> list[tuple[Point, Point]]:
    pl = run.polyline
    return [(pl[i], pl[i + 1]) for i in range(len(pl) - 1)]


def _segment_cells(a: Point, b: Point, cell: float):
    """Yield every grid cell (size ``cell``) the segment ``ab`` passes through.

    A 2D DDA (Amanatides-Woo) grid walk, so a long diagonal registers in
    O(length/cell) cells *along its path* instead of filling its whole bounding
    box (a sheet-spanning diagonal would otherwise create hundreds of thousands
    of entries). Every cell the segment crosses is yielded, so the 3x3 endpoint
    query below stays correct.
    """
    ax, ay = a
    bx, by = b
    cx, cy = math.floor(ax / cell), math.floor(ay / cell)
    ex, ey = math.floor(bx / cell), math.floor(by / cell)
    yield (cx, cy)
    dx, dy = bx - ax, by - ay
    sx = 1 if dx > 0 else -1 if dx < 0 else 0
    sy = 1 if dy > 0 else -1 if dy < 0 else 0
    t_max_x = ((cx + (1 if sx > 0 else 0)) * cell - ax) / dx if sx else math.inf
    t_max_y = ((cy + (1 if sy > 0 else 0)) * cell - ay) / dy if sy else math.inf
    t_delta_x = (cell / abs(dx)) if sx else math.inf
    t_delta_y = (cell / abs(dy)) if sy else math.inf
    # bound the walk to the Manhattan cell span (+slack) so float error can't loop
    remaining = 2 * (abs(ex - cx) + abs(ey - cy)) + 4
    while (cx, cy) != (ex, ey) and remaining > 0:
        remaining -= 1
        if t_max_x <= t_max_y:
            t_max_x += t_delta_x
            cx += sx
        else:
            t_max_y += t_delta_y
            cy += sy
        yield (cx, cy)


def connect_runs(runs: Sequence[Run], *, tol: float) -> list[list[int]]:
    """Group run indices into connected components ("networks").

    Two runs join when an **endpoint of one lands within ``tol`` of any segment
    of the other** — a tee, an elbow, or a small gap at a fitting. Endpoint-to-
    *segment* (not just shared endpoints) is the crux the M5 probe proved: a
    branch tees into the *middle* of a main, so endpoint-only connectivity
    shatters a physically-connected system. ``tol`` is a scale-aware gap bridge
    (see :func:`networks`); a true crossover carries no endpoint at the crossing,
    so a modest ``tol`` does not merge the two lines that cross there.

    Segments are indexed into a ``tol``-sized grid — walked cell-by-cell so a
    long diagonal doesn't fill its bbox — and each endpoint probes only its 3x3
    cell neighborhood, so the pass is near-linear, not O(n^2).
    """
    n = len(runs)
    if n == 0:
        return []
    cell = max(tol, _DEFAULT_TOL)
    grid: dict[tuple[int, int], list[tuple[int, Point, Point]]] = defaultdict(list)
    for i, r in enumerate(runs):
        for a, b in _run_segments(r):
            for c in _segment_cells(a, b, cell):
                grid[c].append((i, a, b))

    uf = _UnionFind(n)
    for i, r in enumerate(runs):
        for p in (r.polyline[0], r.polyline[-1]):
            cx, cy = math.floor(p[0] / cell), math.floor(p[1] / cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j, a, b in grid.get((cx + dx, cy + dy), ()):
                        if j == i or uf.find(i) == uf.find(j):
                            continue
                        if _point_segment_dist(p, a, b) <= tol:
                            uf.union(i, j)

    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)
    return list(comps.values())


def networks(
    runs: Sequence[Run],
    *,
    ppf: float | None = None,
    tol_ft: float = _DEFAULT_NETWORK_TOL_FT,
    tol_pt: float | None = None,
) -> list[Network]:
    """Connect ``runs`` into :class:`Network` s, largest (by length) first.

    The tolerance is **scale-aware**: ``tol_ft`` feet via ``ppf`` (the M5 probe
    put the sweet spot near 0.5 ft — wide enough to bridge the breaks pipe picks
    up at fittings, tight enough that true crossovers don't merge). Pass
    ``tol_pt`` to override in raw points (e.g. when no scale is known). ``runs``
    is whatever candidate set the caller trusts as one discipline's linework;
    connecting across mixed styles is intended (a system may span lineweights).
    """
    if tol_pt is None:
        tol_pt = tol_ft * ppf if ppf else _DEFAULT_TOL
    comps = connect_runs(runs, tol=tol_pt)
    comps.sort(key=lambda c: -sum(runs[k].length_pt for k in c))
    return [
        Network(id=f"N{i}", runs=tuple(runs[k] for k in comp), ppf=ppf)
        for i, comp in enumerate(comps)
    ]


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


# ---------------------------------------------------------------------------
# pipe-size tags (M6): a token -> nominal inches, harvested + snapped to runs
# ---------------------------------------------------------------------------
# Standard fire-protection / plumbing nominal sizes (inches) and their labels.
# A token only counts as a size if it snaps to one of these AND sits next to a
# pipe run (see linear_feet_by_size), so a stray dimension digit is not a size.
_SIZE_LABEL: dict[float, str] = {
    0.5: '1/2"', 0.75: '3/4"', 1.0: '1"', 1.25: '1-1/4"', 1.5: '1-1/2"',
    2.0: '2"', 2.5: '2-1/2"', 3.0: '3"', 4.0: '4"', 6.0: '6"', 8.0: '8"',
}
_UNICODE_FRACTIONS = {
    "¼": 0.25, "½": 0.5, "¾": 0.75, "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
}
_DEFAULT_SIZE_RADIUS_FT = 2.0   # ft: a size tag labels the pipe run within this reach
_SIZE_TRAILING_MARK = re.compile(r'["”Ø]+$')
_ASCII_FRACTION = re.compile(r"(?:(\d+)[-\s])?(\d+)/(\d+)")


def parse_pipe_size_in(text: str) -> float | None:
    """Parse a size token to nominal inches, or ``None`` if it isn't one.

    Handles the notations these sheets actually use — ``1¼`` / ``1½`` (unicode
    fractions), bare ``2`` / ``4``, ``1-1/2`` / ``3/4`` (ascii fractions), with
    an optional ``"`` / ``Ø`` mark — and snaps to a known nominal size, rejecting
    off-grid numbers (a dimension like ``47``, or the ``1/8"`` scale) so only
    real pipe sizes pass.
    """
    s = _SIZE_TRAILING_MARK.sub("", text.strip()).strip()
    if not s:
        return None
    if s[-1] in _UNICODE_FRACTIONS:
        whole = s[:-1].strip()
        try:
            val = (float(whole) if whole else 0.0) + _UNICODE_FRACTIONS[s[-1]]
        except ValueError:
            return None
    elif "/" in s:
        m = _ASCII_FRACTION.fullmatch(s)
        if not m:
            return None
        val = (float(m.group(1)) if m.group(1) else 0.0) + float(m.group(2)) / float(m.group(3))
    else:
        try:
            val = float(s)
        except ValueError:
            return None
    for size in _SIZE_LABEL:
        if abs(val - size) <= 0.02:
            return size
    return None


def size_tags(geometry: SheetGeometry) -> list[tuple[float, Point]]:
    """Every word that parses as a nominal pipe size, as ``(size_in, centroid)``."""
    out: list[tuple[float, Point]] = []
    for w in geometry.words:
        size = parse_pipe_size_in(w.text)
        if size is not None:
            out.append((size, w.centroid))
    return out


def _nearest_size(run: Run, tags: Sequence[tuple[float, Point]], radius: float) -> float | None:
    """Nominal size of the nearest tag within ``radius`` of ``run`` (else ``None``)."""
    segs = _run_segments(run)
    if not segs:
        return None
    best, best_d = None, radius
    for size, c in tags:
        d = min(_point_segment_dist(c, a, b) for a, b in segs)
        if d <= best_d:
            best_d, best = d, size
    return best


def linear_feet_by_size(
    runs: Sequence[Run],
    geometry: SheetGeometry,
    *,
    ppf: float,
    radius_ft: float = _DEFAULT_SIZE_RADIUS_FT,
) -> dict[float | None, float]:
    """Total LF of ``runs`` bucketed by nearest pipe-size tag.

    Each run is attributed the nominal size of the nearest size tag within
    ``radius_ft`` (scale-aware); a run with no tag in reach falls under ``None``
    — the **unsized remainder**, kept first-class because size callouts are
    sparse. Size is a property of the run/segment, not the whole network, so a
    main that reduces shows each size on its own runs.
    """
    tags = size_tags(geometry)
    radius = radius_ft * ppf
    out: dict[float | None, float] = {}
    for r in runs:
        size = _nearest_size(r, tags, radius)
        out[size] = out.get(size, 0.0) + r.length_pt / ppf
    return out


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


def build_networks_report(
    geometry: SheetGeometry,
    *,
    ppf: float | None = None,
    tol_ft: float = _DEFAULT_NETWORK_TOL_FT,
) -> str:
    """M5 report: connect the candidate (heaviest-dark) linework into networks.

    A connected system concentrates its footage in a few networks; a flat spread
    of singletons means connectivity isn't catching the tees/gaps. The candidate
    set is the no-LLM heaviest-dark heuristic — a rough proxy (it can miss branch
    lineweights or catch a matchline), but enough to gate whether connectivity
    holds. A network spanning ~the whole sheet is flagged as a likely matchline.
    """
    if ppf is None:
        ppf = geometry.points_per_foot
    lines = [f"=== M5 networks: {geometry.ref.source} (page {geometry.ref.page_index}) ==="]
    if not ppf:
        return "\n".join(lines + ["  no scale on sheet; cannot compute footage"])

    runs_by = runs_by_style(geometry, ppf=ppf, exclude_border=True)
    # Only consider styles that still have runs: border exclusion can empty a
    # style's list (e.g. a heavy black sheet border), and that empty style must
    # not win the heaviest-dark pick over a thinner pipe pen that has runs.
    pipe_style = heaviest_dark_style(k for k, rs in runs_by.items() if rs)
    pipe = runs_by.get(pipe_style, []) if pipe_style is not None else []
    if not pipe:
        return "\n".join(lines + ["  no candidate (heaviest-dark) linework found"])

    nets = networks(pipe, ppf=ppf, tol_ft=tol_ft)
    total_ft = sum(r.length_pt for r in pipe) / ppf
    pw, ph = geometry.page_width_pt, geometry.page_height_pt
    lines += [
        f"scale: {geometry.scale_label!r} -> {ppf:.4g} pt/ft   "
        f"tolerance: {tol_ft:g} ft ({tol_ft * ppf:.1f} pt)",
        f"candidate lineweight (heaviest dark pen): {_fmt_style(pipe_style)}",
        f"  {len(pipe)} runs, {total_ft:,.1f} LF  ->  {len(nets)} network(s)",
        "",
        f"  {'top networks  (one network = one connected system)':52s} {'runs':>5s} {'LF':>10s} {'%page':>7s}",
    ]
    for nw in nets[:12]:
        x0, y0, x1, y1 = nw.bbox
        pgpct = 100.0 * max((x1 - x0) / pw, (y1 - y0) / ph)
        flag = "  <- spans the sheet (matchline / non-pipe?)" if pgpct >= 80.0 else ""
        lines.append(f"    {nw.id:52s} {nw.run_count:5d} {nw.length_ft:10,.1f} {pgpct:6.0f}%{flag}")
    top3 = sum(nw.length_ft or 0.0 for nw in nets[:3])
    lines += [
        "",
        f"  top-3 networks hold {100.0 * top3 / total_ft:.0f}% of candidate LF "
        f"(a connected system concentrates footage; a flat spread means missed tees/gaps)",
    ]
    return "\n".join(lines)


def build_size_report(
    geometry: SheetGeometry,
    *,
    ppf: float | None = None,
    radius_ft: float = _DEFAULT_SIZE_RADIUS_FT,
) -> str:
    """M6 report: linear feet of the candidate pipe by nominal size.

    The deliverable the tool was started for — "how many LF of each size" — over
    the heaviest-dark candidate linework. Size callouts are sparse, so the
    headline is the total LF with an explicit **unsized remainder**; the size
    split is best-effort over whatever tags snap to a run.
    """
    if ppf is None:
        ppf = geometry.points_per_foot
    lines = [f"=== M6 sizes: {geometry.ref.source} (page {geometry.ref.page_index}) ==="]
    if not ppf:
        return "\n".join(lines + ["  no scale on sheet; cannot compute footage"])

    runs_by = runs_by_style(geometry, ppf=ppf, exclude_border=True)
    pipe_style = heaviest_dark_style(k for k, rs in runs_by.items() if rs)
    pipe = runs_by.get(pipe_style, []) if pipe_style is not None else []
    if not pipe:
        return "\n".join(lines + ["  no candidate (heaviest-dark) linework found"])

    tags = size_tags(geometry)
    by_size = linear_feet_by_size(pipe, geometry, ppf=ppf, radius_ft=radius_ft)
    total = sum(by_size.values())
    unsized = by_size.get(None, 0.0)
    sized = total - unsized
    tag_counts = Counter(_SIZE_LABEL[s] for s, _ in tags)
    lines += [
        f"candidate lineweight (heaviest dark pen): {_fmt_style(pipe_style)}",
        f"  {len(pipe)} runs, {total:,.1f} LF;  size tags found: {len(tags)}"
        + (f"  ({', '.join(f'{k} x{n}' for k, n in tag_counts.most_common())})" if tags else ""),
        "",
        f"LINEAR FEET BY SIZE (nearest size tag within {radius_ft:g} ft):",
    ]
    for size in sorted(s for s in by_size if s is not None):
        lines.append(f"  {_SIZE_LABEL[size]:>8s}  {by_size[size]:>10,.1f} LF")
    pct_unsized = (100.0 * unsized / total) if total else 0.0
    pct_sized = (100.0 * sized / total) if total else 0.0
    lines += [
        f"  {'unsized':>8s}  {unsized:>10,.1f} LF   ({pct_unsized:.0f}% — no size tag within {radius_ft:g} ft)",
        "",
        f"sized {pct_sized:.0f}% of candidate LF "
        f"(size is best-effort; the per-system total stays the trustworthy headline)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="M2 linear measurement / M5 networks report for a vector sheet.")
    ap.add_argument("pdf", help="path to the vector PDF sheet")
    ap.add_argument("--page", type=int, default=0, help="0-based page index (default 0)")
    ap.add_argument("--networks", action="store_true", help="also report M5 connectivity networks")
    ap.add_argument("--sizes", action="store_true", help="also report M6 linear feet by pipe size")
    ap.add_argument(
        "--tol-ft", type=float, default=_DEFAULT_NETWORK_TOL_FT,
        help="network gap tolerance in feet (default %(default)s)",
    )
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args(argv)

    from .geometry import extract_pdf_geometry  # deferred: only the CLI needs PyMuPDF

    geom = extract_pdf_geometry(args.pdf, pages=[args.page])[0]
    report = build_measure_report(geom)
    if args.networks:
        report += "\n\n" + build_networks_report(geom, tol_ft=args.tol_ft)
    if args.sizes:
        report += "\n\n" + build_size_report(geom)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
