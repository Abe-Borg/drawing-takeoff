"""Symbol counting primitives — pure congruence math, no PyMuPDF, no LLM (M9).

Count takeoffs ("how many WCs / sprinkler heads / diffusers") rest on one
property of vector exports: every placement of a CAD block lands on the page as
the SAME geometry, translated (and sometimes rotated or mirrored). So the
engine finds and counts instances deterministically:

  suspects    small-extent paths (a fixture spans a few feet; pipe runs, walls
              and room-spanning hatch lines blow past the cap),
  instances   co-located suspect paths grouped spatially (a flattened block
              arrives from ``get_drawings`` as k loose paths, not one object),
  clusters    instances grouped by a translation-invariant signature of their
              combined geometry, canonicalized under the 8 right-angle
              rotations/mirrors -> exact counts with exact locations.

The LLM never finds or counts anything: :func:`drawing_takeoff.legend.label_symbols`
(M10) only NAMES one exemplar crop per cluster ("Water closet" / "door swing —
not countable"), the same engine-measures / model-names split the length
takeoff rests on. Everything here operates on the backend-free models, so it is
fully unit-testable without a PDF.

Known approximations (documented in DESIGN_COUNTS.md): canonicalization covers
the 8 right-angle orientations — an arbitrary-angle placement lands in its own
cluster and merges later at the component-name level; congruence is same-scale
by design (an enlarged-plan duplicate is a sheet-selection concern, not a
geometry one).

Usage:  python -m drawing_takeoff.count SHEET.pdf [--markup OUT.pdf]
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .measure import _UnionFind
from .models import BBox, GeometryPath, SheetGeometry, StyleKey, SymbolCluster

_DEFAULT_MIN_EXTENT_PT = 2.0    # below this: stipple/dot noise (diagnose's floor)
_DEFAULT_MAX_EXTENT_FT = 4.0    # above this: not a fixture-scale symbol (covers a tub)
_DEFAULT_MAX_EXTENT_PT = 72.0   # cap fallback when the sheet carries no scale
_DEFAULT_JOIN_GAP_PT = 1.5      # bbox proximity that joins paths into one instance
_DEFAULT_MIN_COUNT = 2          # a "repeated symbol" repeats; singletons are summarized
_ROUND_NDIGITS = 1              # 0.1 pt rounding absorbs export float jitter
_GRID_CELL_PT = 36.0            # spatial-index cell for instance grouping

# The 8 congruence transforms — rotations by 90 degrees plus mirrors (the
# dihedral group D4). Orthogonal building geometry places almost every fixture
# in one of these orientations; see DESIGN_COUNTS.md for why arbitrary angles
# are deliberately out of scope.
_DIHEDRAL = (
    lambda x, y: (x, y),
    lambda x, y: (-y, x),
    lambda x, y: (-x, -y),
    lambda x, y: (y, -x),
    lambda x, y: (-x, y),
    lambda x, y: (y, x),
    lambda x, y: (x, -y),
    lambda x, y: (-y, -x),
)


# ---------------------------------------------------------------------------
# congruence signature
# ---------------------------------------------------------------------------
def _transformed_items(path: GeometryPath, t) -> tuple:
    """Path items under point-transform ``t``, every coordinate as a 2-tuple.

    A rectangle is carried as its two corners (re-sorted so min stays first —
    a rotation/mirror otherwise flips which corner is which); unknown ops keep
    their code only, so an exotic segment still participates in equality.
    """
    out = []
    for it in path.items:
        op = it[0]
        if op == "l":
            out.append(("l", t(*it[1]), t(*it[2])))
        elif op == "c":
            out.append(("c", t(*it[1]), t(*it[2]), t(*it[3]), t(*it[4])))
        elif op == "re":
            x0, y0, x1, y1 = it[1]
            (ax, ay), (bx, by) = t(x0, y0), t(x1, y1)
            out.append(("re", (min(ax, bx), min(ay, by)), (max(ax, bx), max(ay, by))))
        elif op == "qu":
            out.append(("qu",) + tuple(t(*p) for p in it[1:5]))
        else:  # pragma: no cover - defensive (geometry normalizes the known ops)
            out.append((op,))
    return tuple(out)


def _shift_round(items: tuple, mx: float, my: float) -> tuple:
    return tuple(
        (it[0],)
        + tuple((round(p[0] - mx, _ROUND_NDIGITS), round(p[1] - my, _ROUND_NDIGITS)) for p in it[1:])
        for it in items
    )


def _style_token(path: GeometryPath) -> str:
    """Stable, orderable style+kind serialization for the signature.

    ``StyleKey`` tuples don't sort when one ``stroke_color`` is ``None``, so
    the signature carries a string. Fill color rides along: two pens drawing
    identical geometry are different symbols.
    """
    return f"{path.style_key!r}|fill={path.fill_color!r}|{path.kind}|closed={path.closed}"


def instance_signature(paths: Sequence[GeometryPath]) -> tuple:
    """Canonical congruence signature of one symbol instance (a set of paths).

    Translation-invariant: coordinates are taken relative to the instance's own
    min corner, so the same block placed anywhere on the page hashes equal.
    Rotation/mirror-canonical: the combined geometry is serialized under each of
    the 8 dihedral transforms and the lexicographically smallest serialization
    wins, so a WC on the opposite wall (rotated or mirrored) joins its
    unrotated siblings. Per-path serializations are sorted, so the export's
    path order can never split a cluster. The normalization is computed over
    ALL member paths together — relative offsets between a symbol's parts are
    part of its identity.

    Coordinates are snapped to the 0.1 pt grid ONCE, up front, and every
    transform after that is exact grid arithmetic — rounding raw floats per
    frame would let a signature derived from a path disagree with the same
    signature derived from a (rounded) core entry, which is precisely the
    equality constellation matching depends on.
    """
    raw = [(_style_token(p), _transformed_items(p, _DIHEDRAL[0])) for p in paths]
    pts = [pt for _tok, items in raw for it in items for pt in it[1:]]
    if not pts:  # pragma: no cover - suspects always carry coordinates
        return ()
    mx = min(p[0] for p in pts)
    my = min(p[1] for p in pts)
    grid = tuple(sorted((tok, _shift_round(items, mx, my)) for tok, items in raw))
    return _canonical_composite(grid)


def signature_hash(signature: tuple) -> str:
    """Short stable hex digest of a signature (the cluster's public identity)."""
    return hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# suspects -> instances -> clusters
# ---------------------------------------------------------------------------
def symbol_suspects(
    geometry: SheetGeometry,
    *,
    ppf: float | None = None,
    min_extent_pt: float = _DEFAULT_MIN_EXTENT_PT,
    max_extent_ft: float = _DEFAULT_MAX_EXTENT_FT,
    max_extent_pt: float | None = None,
) -> tuple[list[GeometryPath], float]:
    """Fixture-scale paths: real extent, but small. Returns ``(paths, cap_pt)``.

    The cap is scale-aware (``max_extent_ft`` via ``ppf``); without a scale it
    falls back to ``_DEFAULT_MAX_EXTENT_PT`` — counting itself never needs the
    scale, only this filter does. Long linework (pipe, walls, room-spanning
    hatch lines, borders) exceeds the cap and never enters the symbol pass.
    """
    if max_extent_pt is not None:
        cap = max_extent_pt
    elif ppf:
        cap = max_extent_ft * ppf
    else:
        cap = _DEFAULT_MAX_EXTENT_PT
    out = [
        p
        for p in geometry.non_degenerate_paths()
        if min_extent_pt <= max(p.bbox_width, p.bbox_height) <= cap
    ]
    return out, cap


# A pen that predominantly draws long linework (pipe runs, walls, leaders) is
# not a symbol pen: its under-cap fragments are stubs and leftovers — the short
# armover a sprinkler head sits on — whose varying lengths fuse into symbol
# instances and shatter their clusters (the FP2.20 probe failure). Both gates
# matter: the share alone would also exclude an annotation pen that draws a few
# long leaders, and the floor alone would exclude nothing on a clean sheet.
_LINEWORK_MIN_LONG = 5      # at least this many over-cap paths, and
_LINEWORK_LONG_SHARE = 0.25  # at least this share of the pen's real paths are long


def linework_styles(
    geometry: SheetGeometry, *, cap_pt: float, min_extent_pt: float = _DEFAULT_MIN_EXTENT_PT
) -> set[StyleKey]:
    """Pens whose paths are predominantly longer than the symbol cap.

    Computed over the whole sheet so the judgment is per-pen, not per-path: the
    pipe pen qualifies because its runs dwarf its stubs; a symbol's outline pen
    does not, even when it also draws the odd long leader.
    """
    long_n: Counter = Counter()
    short_n: Counter = Counter()
    for p in geometry.non_degenerate_paths():
        ext = max(p.bbox_width, p.bbox_height)
        if ext < min_extent_pt:
            continue
        (long_n if ext > cap_pt else short_n)[p.style_key] += 1
    out: set[StyleKey] = set()
    for style, n in long_n.items():
        if n >= _LINEWORK_MIN_LONG and n / (n + short_n.get(style, 0)) >= _LINEWORK_LONG_SHARE:
            out.add(style)
    return out


def _expand(b: BBox, m: float) -> BBox:
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


def _intersects(a: BBox, b: BBox) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def group_instances(
    paths: Sequence[GeometryPath], *, join_gap_pt: float = _DEFAULT_JOIN_GAP_PT
) -> list[list[GeometryPath]]:
    """Group co-located suspect paths into symbol instances.

    Two paths join when their bboxes come within ``join_gap_pt`` of touching —
    a flattened block's parts overlap or abut, while neighboring fixtures keep
    real clearance. Grid-indexed so the pass stays near-linear on a busy sheet.
    """
    n = len(paths)
    if n == 0:
        return []
    half = join_gap_pt / 2.0
    boxes = [_expand(p.bbox, half) for p in paths]
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, b in enumerate(boxes):
        for cx in range(math.floor(b[0] / _GRID_CELL_PT), math.floor(b[2] / _GRID_CELL_PT) + 1):
            for cy in range(math.floor(b[1] / _GRID_CELL_PT), math.floor(b[3] / _GRID_CELL_PT) + 1):
                grid[(cx, cy)].append(i)
    uf = _UnionFind(n)
    for members in grid.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if uf.find(a) != uf.find(b) and _intersects(boxes[a], boxes[b]):
                    uf.union(a, b)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)
    return [[paths[i] for i in comp] for comp in comps.values()]


def _instance_bbox(paths: Sequence[GeometryPath]) -> BBox:
    return (
        min(p.bbox[0] for p in paths), min(p.bbox[1] for p in paths),
        max(p.bbox[2] for p in paths), max(p.bbox[3] for p in paths),
    )


@dataclass
class SymbolScan:
    """Everything the M9 pass learned from one sheet.

    ``clusters`` are the ranked symbol candidates (count >= ``min_count``);
    ``singletons`` are one-off shapes (kept, summarized, never silently
    dropped); ``n_oversized`` counts merged blobs that outgrew the extent cap
    (hatch fields and other connected pattern fill — excluded by geometry, with
    the count surfaced so the exclusion is visible).
    """

    clusters: list[SymbolCluster] = field(default_factory=list)
    singletons: list[SymbolCluster] = field(default_factory=list)
    n_suspects: int = 0
    n_linework: int = 0
    n_unique_parts: int = 0
    n_instances: int = 0
    n_oversized: int = 0
    cap_pt: float = 0.0
    ppf: float | None = None


def scan_symbols(
    geometry: SheetGeometry,
    *,
    ppf: float | None = None,
    min_extent_pt: float = _DEFAULT_MIN_EXTENT_PT,
    max_extent_ft: float = _DEFAULT_MAX_EXTENT_FT,
    max_extent_pt: float | None = None,
    join_gap_pt: float = _DEFAULT_JOIN_GAP_PT,
    min_count: int = _DEFAULT_MIN_COUNT,
) -> SymbolScan:
    """The M9 pass: suspects -> instances -> congruence clusters, ranked.

    Clusters are ranked by count (then drawn size) and assigned ids ``S0…``;
    instances inside a cluster are ordered top-left first so the exemplar is
    deterministic. ``ppf`` defaults to the sheet's own scale and only drives
    the suspect cap and reported sizes — counts themselves are scale-free.
    """
    if ppf is None:
        ppf = geometry.points_per_foot
    suspects, cap = symbol_suspects(
        geometry, ppf=ppf, min_extent_pt=min_extent_pt,
        max_extent_ft=max_extent_ft, max_extent_pt=max_extent_pt,
    )
    scan = SymbolScan(n_suspects=len(suspects), cap_pt=cap, ppf=ppf)

    # Linework pens out: a pipe/wall pen's short fragments are stubs, not
    # symbol parts — the variable armover a head sits on fuses into the head's
    # instance and shatters its cluster by stub length (the FP2.20 probe
    # failure; see DESIGN_COUNTS.md). Judged per pen over the whole sheet.
    lw = linework_styles(geometry, cap_pt=cap, min_extent_pt=min_extent_pt)
    kept = [p for p in suspects if p.style_key not in lw]
    scan.n_linework = len(suspects) - len(kept)

    # Exact duplicates out: CAD exports double-draw symbol parts, and an
    # instance that got 2 copies of an arc must not hash apart from one that
    # got 1 (the M2 stitcher dedups segments for the same reason).
    seen: set = set()
    deduped: list[GeometryPath] = []
    for p in kept:
        key = (_style_token(p), p.items)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    # Stage 1: keep only REPEATED parts. A symbol's parts repeat wherever the
    # symbol repeats, while one-off geometry — leaders, unique text outlines,
    # leftover odd fragments — has a one-off signature and would only fuse
    # neighboring instances apart.
    sigs = [instance_signature([p]) for p in deduped]
    by_sig = Counter(sigs)
    parts = [p for p, s in zip(deduped, sigs) if by_sig[s] >= min_count]
    scan.n_unique_parts = len(suspects) - scan.n_linework - len(parts)

    instances: list[_Instance] = []
    for inst in group_instances(parts, join_gap_pt=join_gap_pt):
        bbox = _instance_bbox(inst)
        if max(bbox[2] - bbox[0], bbox[3] - bbox[1]) > cap:
            scan.n_oversized += 1   # merged pattern blob (hatch field) — not a symbol
            continue
        scan.n_instances += 1
        instances.append(_Instance(bbox=bbox, paths=inst, composite=instance_signature(inst)))

    clusters = _cluster_instances(instances)
    clusters.sort(key=lambda c: (-c.count, -(c.width_pt * c.height_pt)))

    from dataclasses import replace
    for c in clusters:
        if c.count >= min_count:
            scan.clusters.append(replace(c, id=f"S{len(scan.clusters)}"))
        else:
            scan.singletons.append(c)
    return scan


@dataclass
class _Instance:
    """One assembled symbol placement, with everything clustering needs."""

    bbox: BBox
    paths: list[GeometryPath]
    composite: tuple          # position-aware canonical signature of the whole

    @property
    def anchor(self) -> tuple:
        """The most distinctive member part — largest drawn area, ties broken
        deterministically. The family key for core extraction."""
        def rank(p: GeometryPath):
            sig = instance_signature([p])
            return (round(p.bbox_width * p.bbox_height, 1),
                    round(max(p.bbox_width, p.bbox_height), 1), repr(sig)), sig
        return max((rank(p) for p in self.paths), key=lambda rs: rs[0])[1]


def _transform_positioned(comp: tuple, t) -> tuple:
    """Re-canonicalize a positioned composite under one dihedral transform.

    A composite's frame is whichever of the 8 transforms serialized smallest —
    and co-located annotation can tip that pick, so two drawings of one symbol
    may canonicalize into different frames. Core intersection and membership
    therefore compare composites across all 8 relative transforms.
    """
    pairs = []
    pts: list[tuple[float, float]] = []
    for tok, items in comp:
        titems = _titems(items, t)
        pairs.append((tok, titems))
        for it in titems:
            pts.extend(it[1:])
    mx = min(p[0] for p in pts)
    my = min(p[1] for p in pts)
    return tuple(sorted((tok, _shift_round(items, mx, my)) for tok, items in pairs))


def _titems(items: tuple, t) -> tuple:
    """Positioned items under transform ``t`` (rect corners re-sorted, as ever)."""
    out = []
    for it in items:
        if it[0] == "re":
            (ax, ay), (bx, by) = t(*it[1]), t(*it[2])
            out.append(("re", (min(ax, bx), min(ay, by)), (max(ax, bx), max(ay, by))))
        else:
            out.append((it[0],) + tuple(t(*p) for p in it[1:]))
    return tuple(out)


def _canonical_composite(comp: tuple) -> tuple:
    """Frame-canonical form of a positioned composite (min over the 8 frames)."""
    return min(_transform_positioned(comp, t) for t in _DIHEDRAL)


def _entries(comp: tuple) -> Counter:
    """A positioned composite as a multiset of (style, positioned-items) parts."""
    return Counter(comp)


# Two repeated composites describe the same symbol when their frame-aligned
# intersection keeps at least this share of the larger one's INK — co-located
# annotation (a tiny tick) trims a little, a genuinely different symbol (a
# bare anchor look-alike, a structurally different variant) trims a lot. Ink
# is extent-weighted so a 9 pt disc outvotes a 2 pt fragment regardless of how
# many entries each contributes.
_CORE_SIMILARITY = 0.7


def _entry_weight(entry: tuple) -> float:
    """Ink weight of one positioned part: (its extent + epsilon) squared."""
    _tok, items = entry
    pts = [p for it in items for p in it[1:]]
    dx = max(p[0] for p in pts) - min(p[0] for p in pts)
    dy = max(p[1] for p in pts) - min(p[1] for p in pts)
    return (max(dx, dy) + 0.1) ** 2


def _ink(entries: Counter) -> float:
    return sum(_entry_weight(e) * n for e, n in entries.items())


def _aligned_intersection(a: Counter, comp: tuple) -> Counter:
    """Heaviest intersection of ``a`` with ``comp`` across the 8 relative frames."""
    best: Counter | None = None
    best_ink = -1.0
    for t in _DIHEDRAL:
        cand = a & _entries(_transform_positioned(comp, t))
        ink = _ink(cand)
        if ink > best_ink:
            best, best_ink = cand, ink
    return best if best is not None else Counter()


def _family_cores(repeated: list[tuple[tuple, int]]) -> list[Counter]:
    """One positioned core per group of similar repeated composites.

    STAR grouping, not transitive chaining: composites are taken most-repeated
    first, and each joins the first group whose REPRESENTATIVE (its founding,
    most-repeated composite) it shares >= 70% of ink with — a chain of
    pairwise-similar variants can otherwise link two genuinely different
    symbols and intersect their core down to a bare shared part. Each group's
    core is the fold of aligned intersections into the representative's frame.
    A family can carry several cores (head-with-tick variants fold into one; a
    bare-disc look-alike that repeats keeps its own).
    """
    groups: list[dict] = []
    ordered = sorted(
        repeated,
        key=lambda cn: (-cn[1], -_ink(_entries(cn[0]))),
    )
    for comp, _n in ordered:
        e = _entries(comp)
        ink_e = _ink(e)
        for g in groups:
            shared = _ink(_aligned_intersection(g["rep"], comp))
            if shared >= _CORE_SIMILARITY * max(g["rep_ink"], ink_e):
                g["core"] = _aligned_intersection(g["core"], comp)
                break
        else:
            groups.append({"rep": e, "rep_ink": ink_e, "core": e})
    return [g["core"] for g in groups if g["core"]]


# Tolerances for constellation matching, in points. Coordinates compared here
# were rounded relative to DIFFERENT origins (a path's own corner vs. its
# instance's corner), so equality must allow the two rounding steps to
# disagree; both bounds stay far tighter than any real part spacing.
_MATCH_TOL = 0.35        # placement: where a part sits within the constellation
_GEOM_TOL = 0.25         # shape: per-coordinate agreement of normalized items


@dataclass
class _MatchPart:
    """One matchable part — an instance path, or a core entry during tiling."""

    token: str                       # style+kind serialization (must match exactly)
    norm: tuple                      # items normalized to the part's own min corner
    bbox: BBox
    style_key: StyleKey | None


def _items_bbox(items: tuple) -> BBox:
    pts = [p for it in items for p in it[1:]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _own_norm(items: tuple) -> tuple:
    bb = _items_bbox(items)
    return _shift_round(items, bb[0], bb[1])


def _part_of(p: GeometryPath) -> _MatchPart:
    items = _transformed_items(p, _DIHEDRAL[0])
    return _MatchPart(
        token=_style_token(p), norm=_own_norm(items), bbox=p.bbox, style_key=p.style_key,
    )


def _items_close(a: tuple, b: tuple, tol: float = _GEOM_TOL) -> bool:
    """Whether two own-normalized item tuples draw the same shape (within tol).

    Item ORDER is preserved through every serialization (only whole paths get
    sorted), so this is a positional walk, not a matching problem.
    """
    if len(a) != len(b):
        return False
    for ia, ib in zip(a, b):
        if ia[0] != ib[0] or len(ia) != len(ib):
            return False
        for pa, pb in zip(ia[1:], ib[1:]):
            if abs(pa[0] - pb[0]) > tol or abs(pa[1] - pb[1]) > tol:
                return False
    return True


def _bbox_under(b: BBox, t) -> BBox:
    corners = [t(b[0], b[1]), t(b[2], b[1]), t(b[0], b[3]), t(b[2], b[3])]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class _Core:
    """A discovered symbol shape, ready for constellation matching.

    ``frames[k]`` holds, per entry (heaviest part first), the entry's token,
    own-normalized items, and bbox — all under dihedral transform ``k`` — so
    the match loop does no per-attempt geometry work.
    """

    frames: list[list[tuple[str, tuple, BBox]]]
    ink: float
    canonical: tuple                 # frame-canonical composite (identity + hash)


def _build_core(entries: Counter) -> _Core:
    expanded = []
    for e, n in entries.items():
        for _ in range(n):
            expanded.append(e)
    expanded.sort(key=lambda e: -_entry_weight(e))
    frames = []
    for t in _DIHEDRAL:
        frame = []
        for tok, items in expanded:
            titems = _titems(items, t)
            frame.append((tok, _own_norm(titems), _items_bbox(titems)))
        frames.append(frame)
    return _Core(
        frames=frames,
        ink=_ink(entries),
        canonical=_canonical_composite(tuple(sorted(entries.elements()))),
    )


def _match_core(parts: list[_MatchPart], free: set[int], core: _Core) -> list[int] | None:
    """Find one placement of ``core`` among the free ``parts``, or ``None``.

    Constellation matching anchored on the core's heaviest part: each candidate
    anchor path fixes a translation per frame, and every core part must then
    sit at its implied position (within ``_MATCH_TOL``) with the same style
    token and the same shape (within ``_GEOM_TOL``). Subset matching, so
    co-located extras (the tick, an adjacent label) are simply left unconsumed.
    """
    for k, frame in enumerate(core.frames):
        a_tok, a_norm, a_bb = frame[0]
        for pi in sorted(free):
            p = parts[pi]
            if p.token != a_tok or not _items_close(p.norm, a_norm):
                continue
            dx = p.bbox[0] - a_bb[0]
            dy = p.bbox[1] - a_bb[1]
            used: list[int] = []
            taken: set[int] = set()
            ok = True
            for tok, norm, bb in frame:
                ex, ey = bb[0] + dx, bb[1] + dy
                cand = None
                for j in free:
                    q = parts[j]
                    if j in taken or q.token != tok:
                        continue
                    if abs(q.bbox[0] - ex) > _MATCH_TOL or abs(q.bbox[1] - ey) > _MATCH_TOL:
                        continue
                    if _items_close(q.norm, norm):
                        cand = j
                        break
                if cand is None:
                    ok = False
                    break
                used.append(cand)
                taken.add(cand)
            if ok:
                return used
    return None


def _fully_tiled(core: _Core, smaller: list[_Core]) -> bool:
    """Whether ``core`` is a CO-LOCATION of smaller cores rather than a symbol.

    Two conditions, both required. (a) Exact tiling: smaller cores consume
    every part. (b) Spatial separability: the consumed pieces form >= 2
    disjoint groups — a head fused with its size label tiles into pieces that
    sit apart, while a real symbol's own parts nest/overlap, so a 12-part head
    "tiled" by 12 position-free single-part atoms still forms ONE overlapping
    group and is kept. Partial tiling keeps the core whole: conservative,
    never silently lossy.
    """
    parts = [
        _MatchPart(token=tok, norm=norm, bbox=bb, style_key=None)
        for tok, norm, bb in core.frames[0]
    ]
    free = set(range(len(parts)))
    piece_boxes: list[BBox] = []
    for atom in smaller:
        while free:
            hit = _match_core(parts, free, atom)
            if hit is None:
                break
            free -= set(hit)
            piece_boxes.append((
                min(parts[j].bbox[0] for j in hit), min(parts[j].bbox[1] for j in hit),
                max(parts[j].bbox[2] for j in hit), max(parts[j].bbox[3] for j in hit),
            ))
    if free or len(piece_boxes) < 2:
        return False
    uf = _UnionFind(len(piece_boxes))
    grown = [_expand(b, 0.5) for b in piece_boxes]
    for i in range(len(grown)):
        for j in range(i + 1, len(grown)):
            if _intersects(grown[i], grown[j]):
                uf.union(i, j)
    return len({uf.find(i) for i in range(len(grown))}) >= 2


def _make_cluster(members: list[_Instance], *, signature, variants: int) -> SymbolCluster:
    """Build a cluster from whole leftover instances (exact-composite group)."""
    members = sorted(members, key=lambda m: (m.bbox[1], m.bbox[0]))
    ex = members[0]
    return SymbolCluster(
        id="",
        signature_hash=signature_hash(signature),
        instance_bboxes=tuple(m.bbox for m in members),
        paths_per_instance=len(ex.paths),
        width_pt=ex.bbox[2] - ex.bbox[0],
        height_pt=ex.bbox[3] - ex.bbox[1],
        style_keys=tuple(dict.fromkeys(p.style_key for p in ex.paths)),
        variants=variants,
    )


def _cluster_from_occurrences(core: _Core, occs: list[tuple]) -> SymbolCluster:
    """Build a cluster from matched core occurrences.

    The exemplar is the top-left occurrence from the MODAL source arrangement,
    so the M10 crop shows the typical drawing of the symbol, not an outlier.
    """
    comps = Counter(o[2] for o in occs)
    modal = comps.most_common(1)[0][0]
    occs = sorted(occs, key=lambda o: (o[2] != modal, o[0][1], o[0][0]))
    bb0 = occs[0][0]
    return SymbolCluster(
        id="",
        signature_hash=signature_hash(core.canonical),
        instance_bboxes=tuple(o[0] for o in occs),
        paths_per_instance=len(core.frames[0]),
        width_pt=bb0[2] - bb0[0],
        height_pt=bb0[3] - bb0[1],
        style_keys=occs[0][1],
        variants=len(comps),
    )


def _cluster_instances(instances: list[_Instance]) -> list[SymbolCluster]:
    """Core discovery + occurrence counting.

    Exact composite equality alone shatters real symbols: annotation that
    co-locates with a symbol but isn't part of its block (the FP2.20 probe's
    pipe-connection tick, 1-3 copies in varying spots inside each head's bbox)
    varies per placement, splitting 99 congruent heads into 69 composites —
    and a symbol fused with a neighbor (a head under its size label) hides in
    a composite anchored on the neighbor. So:

      discover  anchor families (instances sharing their most distinctive
                part) propose cores: the positioned, frame-aligned
                intersection of each group of mutually-similar repeated
                composites (see :func:`_family_cores`) — block parts sit at
                the same canonical offsets everywhere and survive; the
                wandering tick falls out,
      atomize   a core that tiles exactly into smaller cores is a co-location,
                not a symbol — dropped, so its constituents are credited,
      count     every instance is a bag of co-located symbols: cores are
                constellation-matched into it heaviest-first, consuming the
                matched paths, so one fused instance credits the head AND the
                label, and a bare anchor look-alike (a lone disc) matches no
                core and falls through to exact-composite grouping.

    ``variants`` records how many distinct source arrangements fed a cluster;
    multi-variant clusters earn a review note (two genuinely different symbols
    sharing one full core would merge here — visible, never silent; see
    DESIGN_COUNTS.md).
    """
    families: dict[tuple, list[_Instance]] = defaultdict(list)
    for inst in instances:
        families[inst.anchor].append(inst)

    by_canon: dict[tuple, Counter] = {}
    for fam in families.values():
        by_comp = Counter(m.composite for m in fam)
        for entries in _family_cores([(c, n) for c, n in by_comp.items() if n >= 2]):
            canon = _canonical_composite(tuple(sorted(entries.elements())))
            by_canon.setdefault(canon, entries)

    cores = [_build_core(entries) for entries in by_canon.values()]
    cores.sort(key=lambda c: -c.ink)
    cores = [
        c for i, c in enumerate(cores)
        if not _fully_tiled(c, [s for s in cores if s.ink < c.ink * 0.999])
    ]

    occs: list[list[tuple]] = [[] for _ in cores]
    leftovers: list[_Instance] = []
    for inst in instances:
        parts = [_part_of(p) for p in inst.paths]
        token_set = {p.token for p in parts}
        free = set(range(len(parts)))
        matched = False
        for ci, core in enumerate(cores):   # heaviest first: most specific symbol wins its parts
            if core.frames[0][0][0] not in token_set:
                continue   # anchor pen absent — cheap skip before constellation work
            while free:
                hit = _match_core(parts, free, core)
                if hit is None:
                    break
                matched = True
                free -= set(hit)
                bb = (
                    min(parts[j].bbox[0] for j in hit), min(parts[j].bbox[1] for j in hit),
                    max(parts[j].bbox[2] for j in hit), max(parts[j].bbox[3] for j in hit),
                )
                styles = tuple(dict.fromkeys(parts[j].style_key for j in hit))
                occs[ci].append((bb, styles, inst.composite))
        if not matched:
            leftovers.append(inst)

    clusters = [
        _cluster_from_occurrences(core, olist) for core, olist in zip(cores, occs) if olist
    ]
    by_exact: dict[tuple, list[_Instance]] = defaultdict(list)
    for m in leftovers:
        by_exact[m.composite].append(m)
    for comp, members in by_exact.items():
        clusters.append(_make_cluster(members, signature=comp, variants=1))
    return clusters


# ---------------------------------------------------------------------------
# fixture-tag cross-check (advisory)
# ---------------------------------------------------------------------------
# Fixture/equipment tags as drawn: 1-5 capitals, optional dash, 1-3 digits
# ("WC-1", "LAV2", "P-1", "FD-3"). Pure numbers and pipe sizes never match
# (no leading letters), so the length takeoff's tokens stay out.
_TAG_RE = re.compile(r"[A-Z]{1,5}-?\d{1,3}")


def fixture_tag_counts(geometry: SheetGeometry) -> Counter:
    """Occurrences of fixture-tag-shaped text, by tag.

    An independent, exact signal from the drawing's own annotation: geometry
    counts and tag counts should roughly agree (the M1 earned-agreement move).
    Advisory only — tags are sparse and inconsistent, so a mismatch flags a
    human glance, never a silent adjustment of either number.
    """
    out: Counter = Counter()
    for w in geometry.words:
        text = w.text.strip()
        if _TAG_RE.fullmatch(text):
            out[text] += 1
    return out


# ---------------------------------------------------------------------------
# M9 report + CLI (no LLM)
# ---------------------------------------------------------------------------
def _fmt_size(c: SymbolCluster, ppf: float | None) -> str:
    if ppf:
        return f"{c.width_pt / ppf:.1f}x{c.height_pt / ppf:.1f} ft"
    return f"{c.width_pt:.0f}x{c.height_pt:.0f} pt"


def _fmt_style_keys(c: SymbolCluster) -> str:
    parts = []
    for k in c.style_keys[:2]:
        col = "none" if k.stroke_color is None else ",".join(f"{v:.2f}" for v in k.stroke_color)
        parts.append(f"[{col}] w={k.width}")
    if len(c.style_keys) > 2:
        parts.append(f"+{len(c.style_keys) - 2} more")
    return "; ".join(parts)


def scan_review_notes(scan: SymbolScan, *, top: int) -> list[str]:
    """What the counts pass did NOT take to the model — surfaced, never hidden.

    Mirrors the System×Size review discipline: clusters beyond the top-N cap,
    one-off shapes below the repeat floor, and merged pattern blobs over the
    extent cap each get one honest line.
    """
    notes: list[str] = []
    remainder = scan.clusters[top:]
    if remainder:
        notes.append(
            f"NOT REVIEWED: {len(remainder)} smaller clusters beyond the top {top} = "
            f"{sum(c.count for c in remainder)} instances (raise the cluster cap to include them)."
        )
    if scan.singletons or scan.n_unique_parts:
        notes.append(
            f"SINGLETONS: {len(scan.singletons)} one-off arrangements and {scan.n_unique_parts} "
            "one-off paths below the repeat floor (a unique fixture lands here — check the tag "
            "cross-check; --min-count 1 to include everything)."
        )
    if scan.n_oversized:
        notes.append(
            f"EXCLUDED as oversized pattern blobs: {scan.n_oversized} merged groups over the "
            f"{scan.cap_pt:.0f} pt extent cap (hatch/pattern fill)."
        )
    return notes


def build_scan_report(geometry: SheetGeometry, scan: SymbolScan, *, top: int = 12) -> str:
    """M9 deliverable: the congruence-cluster table, sized and ranked, plus the
    advisory tag cross-check — the go/no-go evidence that fixtures really do
    export as countable congruent geometry."""
    lines = [
        f"=== M9 symbol clusters: {geometry.ref.source} (page {geometry.ref.page_index}) ===",
        f"scale: {geometry.scale_label or 'unknown'}"
        + (f"  ->  {scan.ppf:g} pt/ft" if scan.ppf else "  (counts are scale-free; sizes in pt)"),
        f"suspects (extent {_DEFAULT_MIN_EXTENT_PT:g}-{scan.cap_pt:.0f} pt): {scan.n_suspects} paths; "
        f"dropped {scan.n_linework} linework-pen fragments + {scan.n_unique_parts} one-off parts "
        f"-> {scan.n_instances} instances -> {len(scan.clusters)} repeated shapes "
        f"(+{len(scan.singletons)} singleton arrangements, {scan.n_oversized} oversized blobs excluded)",
        "",
        f"  {'top clusters (count = congruent instances)':44s} {'count':>6s} {'size':>12s}  {'paths':>5s} {'vars':>4s}  style",
    ]
    for c in scan.clusters[:top]:
        lines.append(
            f"    {c.id:42s} {c.count:6d} {_fmt_size(c, scan.ppf):>12s}  {c.paths_per_instance:5d} {c.variants:4d}  "
            f"{_fmt_style_keys(c)}"
        )
    notes = scan_review_notes(scan, top=top)
    if notes:
        lines += [""] + [f"  {n}" for n in notes]
    tags = fixture_tag_counts(geometry).most_common(12)
    if tags:
        lines += [
            "",
            "  ADVISORY tag cross-check (fixture-tag-shaped text on the sheet): "
            + ", ".join(f"{t} x{n}" for t, n in tags),
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="M9 symbol congruence scan (no LLM): repeated-shape clusters with exact counts."
    )
    ap.add_argument("pdf", help="path to the vector PDF sheet")
    ap.add_argument("--page", type=int, default=0, help="0-based page index (default 0)")
    ap.add_argument("--top", type=int, default=12, help="clusters to show (default %(default)s)")
    ap.add_argument("--min-count", type=int, default=_DEFAULT_MIN_COUNT,
                    help="repeat floor for a cluster (default %(default)s; 1 includes one-offs)")
    ap.add_argument("--max-extent-ft", type=float, default=_DEFAULT_MAX_EXTENT_FT,
                    help="suspect extent cap in feet at sheet scale (default %(default)s)")
    ap.add_argument("--markup", default=None,
                    help="also write a PDF with every instance of the top clusters boxed + labeled")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args(argv)

    from .geometry import extract_pdf_geometry  # deferred: only the CLI needs PyMuPDF

    geom = extract_pdf_geometry(args.pdf, pages=[args.page])[0]
    scan = scan_symbols(geom, min_count=args.min_count, max_extent_ft=args.max_extent_ft)
    report = build_scan_report(geom, scan, top=args.top)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
    if args.markup:
        from .geometry import write_count_markup_pdf

        write_count_markup_pdf(args.pdf, args.page, scan.clusters[: args.top], args.markup)
        print(f"\nWrote instance markup: {args.markup}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
