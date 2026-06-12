"""Hermetic tests for the M9 symbol congruence pass (no PDF, no PyMuPDF, no LLM).

Synthetic ``GeometryPath`` sets exercise the whole suspects -> instances ->
clusters path: translation invariance, rotation/mirror canonicalization, path-
order independence, the extent cap, instance grouping, the oversized-blob
exclusion, and the advisory fixture-tag cross-check.
"""
from __future__ import annotations

import pytest

from drawing_takeoff import count
from drawing_takeoff.models import GeometryPath, SheetGeometry, SheetRef, TextWord


def _bbox_of(items):
    xs, ys = [], []
    for it in items:
        if it[0] == "re":
            x0, y0, x1, y1 = it[1]
            xs += [x0, x1]
            ys += [y0, y1]
        else:
            for p in it[1:]:
                xs.append(p[0])
                ys.append(p[1])
    return (min(xs), min(ys), max(xs), max(ys))


def _path(items, *, color=(0.0, 0.0, 0.0), width=0.7, fill=None, kind="stroke", dashes="[] 0"):
    return GeometryPath(
        items=tuple(items), stroke_color=color, fill_color=fill, width=width,
        dashes=dashes, closed=False, bbox=_bbox_of(items), kind=kind,
    )


def _translate(items, dx, dy):
    out = []
    for it in items:
        if it[0] == "re":
            x0, y0, x1, y1 = it[1]
            out.append(("re", (x0 + dx, y0 + dy, x1 + dx, y1 + dy)))
        else:
            out.append((it[0],) + tuple((p[0] + dx, p[1] + dy) for p in it[1:]))
    return out


# An asymmetric L-shaped two-path symbol (arms 10 and 6) — unequal arms make the
# rotation/mirror assertions meaningful.
_ARM_A = [("l", (0.0, 0.0), (10.0, 0.0))]
_ARM_B = [("l", (0.0, 0.0), (0.0, 6.0))]


def _symbol_at(dx, dy):
    return [_path(_translate(_ARM_A, dx, dy)), _path(_translate(_ARM_B, dx, dy))]


# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------
def test_signature_is_translation_invariant():
    assert count.instance_signature(_symbol_at(0, 0)) == count.instance_signature(_symbol_at(300, 150))


def test_signature_canonicalizes_rotation_and_mirror():
    base = count.instance_signature(_symbol_at(0, 0))
    # Rotated 90 degrees ((x, y) -> (-y, x), then translated back on-page).
    rot = [
        _path([("l", (100.0, 100.0), (100.0, 110.0))]),
        _path([("l", (100.0, 100.0), (94.0, 100.0))]),
    ]
    # Mirrored ((x, y) -> (-x, y), translated).
    mir = [
        _path([("l", (50.0, 50.0), (40.0, 50.0))]),
        _path([("l", (50.0, 50.0), (50.0, 56.0))]),
    ]
    assert count.instance_signature(rot) == base
    assert count.instance_signature(mir) == base


def test_signature_distinguishes_different_geometry_and_pens():
    base = count.instance_signature(_symbol_at(0, 0))
    longer = [_path([("l", (0.0, 0.0), (12.0, 0.0))]), _path(_ARM_B)]
    other_pen = [_path(_ARM_A, width=1.4), _path(_ARM_B)]
    assert count.instance_signature(longer) != base
    assert count.instance_signature(other_pen) != base


def test_signature_is_path_order_independent():
    a, b = _symbol_at(0, 0)
    assert count.instance_signature([a, b]) == count.instance_signature([b, a])


def test_signature_merges_rotated_rect():
    # A 10x6 rect and its 90-degree rotation (6x10) are the same drawn shape.
    r1 = _path([("re", (0.0, 0.0, 10.0, 6.0))])
    r2 = _path([("re", (80.0, 20.0, 86.0, 30.0))])
    assert count.instance_signature([r1]) == count.instance_signature([r2])


def test_signature_handles_beziers():
    c1 = _path([("c", (0.0, 0.0), (3.0, 4.0), (7.0, 4.0), (10.0, 0.0))])
    c2 = _path([("c", (60.0, 90.0), (63.0, 94.0), (67.0, 94.0), (70.0, 90.0))])
    assert count.instance_signature([c1]) == count.instance_signature([c2])


# ---------------------------------------------------------------------------
# suspects + instance grouping
# ---------------------------------------------------------------------------
def _geom(paths, *, words=(), ppf=9.0):
    return SheetGeometry(
        ref=SheetRef("A1.pdf", 0), page_width_pt=612, page_height_pt=396,
        paths=list(paths), words=list(words),
        scale_label='1/8" = 1\'-0"' if ppf else None, points_per_foot=ppf,
    )


def test_suspects_cap_is_scale_aware_and_drops_long_linework():
    fixture = _path(_ARM_A)                                  # 10 pt — in
    pipe = _path([("l", (0.0, 200.0), (200.0, 200.0))])      # 200 pt — out
    dot = _path([("l", (5.0, 5.0), (5.5, 5.0))])             # 0.5 pt — stipple, out
    suspects, cap = count.symbol_suspects(_geom([fixture, pipe, dot]), ppf=9.0)
    assert cap == pytest.approx(36.0)                        # 4 ft at 9 pt/ft
    assert suspects == [fixture]


def test_suspects_fall_back_to_point_cap_without_scale():
    _, cap = count.symbol_suspects(_geom([], ppf=None), ppf=None)
    assert cap == pytest.approx(count._DEFAULT_MAX_EXTENT_PT)


def test_group_instances_joins_near_and_keeps_separated():
    near_a = _path([("l", (0.0, 0.0), (10.0, 0.0))])
    near_b = _path([("l", (10.5, 0.0), (10.5, 6.0))])        # 0.5 pt gap -> joins
    far = _path([("l", (200.0, 200.0), (210.0, 200.0))])
    groups = count.group_instances([near_a, near_b, far])
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]


# ---------------------------------------------------------------------------
# scan_symbols end to end
# ---------------------------------------------------------------------------
def test_scan_clusters_rank_and_count_with_rotated_instance():
    paths = (
        _symbol_at(0, 0) + _symbol_at(100, 0)
        # third instance of the L, rotated 90 degrees:
        + [_path([("l", (300.0, 100.0), (300.0, 110.0))]),
           _path([("l", (300.0, 100.0), (294.0, 100.0))])]
        # a second, smaller symbol, twice:
        + [_path([("re", (0.0, 50.0, 4.0, 54.0))]), _path([("re", (40.0, 50.0, 44.0, 54.0))])]
        # a one-off shape (dropped as a one-off part) and a long pipe run:
        + [_path([("l", (500.0, 300.0), (508.0, 305.0))]),
           _path([("l", (0.0, 390.0), (600.0, 390.0))], width=1.3)]
    )
    scan = count.scan_symbols(_geom(paths))
    assert [c.count for c in scan.clusters] == [3, 2]
    assert scan.clusters[0].id == "S0" and scan.clusters[1].id == "S1"
    assert scan.clusters[0].paths_per_instance == 2
    assert len(scan.clusters[0].instance_bboxes) == 3
    assert scan.n_suspects == 9          # the pipe run never enters the pass
    assert scan.n_unique_parts == 1      # the one-off shape drops at stage 1
    assert scan.n_instances == 3 + 2


def test_linework_pen_stub_cannot_shatter_a_symbol_cluster():
    # The FP2.20 head anatomy: a symbol in its own pens, touched by short stubs
    # of the PIPE pen whose lengths repeat (so stage 1 alone keeps them). The
    # pipe pen draws mostly long runs, so the per-pen linework gate must drop
    # its fragments and the heads must land in ONE cluster.
    pipe = dict(color=(0.0, 0.0, 0.0), width=1.3)
    runs = [_path([("l", (0.0, float(360 + i * 4)), (600.0, float(360 + i * 4)))], **pipe)
            for i in range(5)]
    heads = _symbol_at(0, 0) + _symbol_at(100, 0) + _symbol_at(200, 0)
    stubs = [  # repeated stub lengths: 18, 18, 11 — stage 1 keeps the 18s
        _path([("l", (10.0, 0.0), (28.0, 0.0))], **pipe),
        _path([("l", (110.0, 0.0), (128.0, 0.0))], **pipe),
        _path([("l", (210.0, 0.0), (221.0, 0.0))], **pipe),
    ]
    scan = count.scan_symbols(_geom(runs + heads + stubs))
    assert scan.n_linework == 3
    assert [c.count for c in scan.clusters] == [3]
    assert scan.clusters[0].paths_per_instance == 2


def test_varying_co_located_annotation_does_not_shatter_a_family():
    # The FP2.20 donut failure in miniature: every head has the same core (a
    # rect "disc" + a line), plus 0-2 tick fragments at varying spots INSIDE
    # the bbox. Exact composites differ; the anchor family's core must absorb
    # the variants into ONE cluster with `variants` > 1.
    def head(dx, dy, ticks):
        ps = [
            _path([("re", (dx, dy, dx + 9.0, dy + 9.0))]),
            _path([("l", (dx + 1.0, dy + 4.5), (dx + 8.0, dy + 4.5))]),
        ]
        for tx in ticks:   # tick: a small repeated fragment at a varying offset
            ps.append(_path([("l", (dx + tx, dy + 1.0), (dx + tx, dy + 3.5))], width=0.36))
        return ps

    paths = (
        head(0, 0, [2.0]) + head(50, 0, [2.0])          # repeated composite A
        + head(100, 0, [6.0]) + head(150, 0, [6.0])     # repeated composite B
        + head(200, 0, [2.0, 6.0])                      # one-off variant C
        + head(250, 0, [])                              # bare core
    )
    scan = count.scan_symbols(_geom(paths))
    assert [c.count for c in scan.clusters] == [6]
    assert scan.clusters[0].variants == 4
    assert scan.clusters[0].paths_per_instance == 2   # the CORE: rect + line, ticks out


def test_anchor_lookalike_without_the_core_stays_separate():
    # A lone disc (the family anchor) is NOT a head: core = disc+line from the
    # repeated composites, so the bare disc falls back to its own grouping.
    def head(dx, dy):
        return [
            _path([("re", (dx, dy, dx + 9.0, dy + 9.0))]),
            _path([("l", (dx + 1.0, dy + 4.5), (dx + 8.0, dy + 4.5))]),
        ]
    discs = [_path([("re", (200.0, 50.0, 209.0, 59.0))]),
             _path([("re", (250.0, 50.0, 259.0, 59.0))])]
    scan = count.scan_symbols(_geom(head(0, 0) + head(50, 0) + discs))
    assert sorted(c.count for c in scan.clusters) == [2, 2]
    head_cluster = next(c for c in scan.clusters if c.paths_per_instance == 2)
    disc_cluster = next(c for c in scan.clusters if c.paths_per_instance == 1)
    assert head_cluster.count == 2 and disc_cluster.count == 2


def test_exact_duplicate_paths_do_not_split_a_cluster():
    # One instance double-draws an arm (same pen, same coords): it must hash
    # with the clean instances, not become its own shape.
    paths = _symbol_at(0, 0) + _symbol_at(100, 0) + [_path(_translate(_ARM_A, 100, 0))]
    scan = count.scan_symbols(_geom(paths))
    assert [c.count for c in scan.clusters] == [2]


def test_one_off_part_cannot_fuse_a_symbol_instance_apart():
    # The FP2.20 probe failure: a head-like symbol sits at the end of a
    # variable-length stub whose bbox touches it. The stubs are one-off
    # geometry, so stage 1 must drop them and the three heads must still land
    # in ONE cluster — without it, head+stub composites split by stub length.
    heads = _symbol_at(0, 0) + _symbol_at(100, 0) + _symbol_at(200, 0)
    stubs = [
        _path([("l", (10.0, 0.0), (28.0, 0.0))]),    # touches instance 1, 18 pt
        _path([("l", (110.0, 0.0), (121.0, 0.0))]),  # touches instance 2, 11 pt
        _path([("l", (210.0, 0.0), (235.0, 0.0))]),  # touches instance 3, 25 pt
    ]
    scan = count.scan_symbols(_geom(heads + stubs))
    assert [c.count for c in scan.clusters] == [3]
    assert scan.n_unique_parts == 3


def test_scan_min_count_one_includes_singletons():
    paths = _symbol_at(0, 0) + [_path([("l", (500.0, 300.0), (508.0, 305.0))])]
    scan = count.scan_symbols(_geom(paths), min_count=1)
    assert not scan.singletons
    assert sorted(c.count for c in scan.clusters) == [1, 1]
    assert {c.id for c in scan.clusters} == {"S0", "S1"}


def test_scan_excludes_oversized_merged_blob():
    # Six congruent short strokes overlapping in a chain: individually under the
    # cap, merged into one 50 pt blob over it -> excluded as pattern fill, with
    # the exclusion surfaced, and NOT counted as a 6-instance cluster.
    blob = [_path([("l", (float(i * 8), 0.0), (float(i * 8 + 10), 10.0))]) for i in range(6)]
    scan = count.scan_symbols(_geom(blob))
    assert scan.n_oversized == 1
    assert scan.clusters == [] and scan.singletons == []
    assert any("oversized" in n for n in count.scan_review_notes(scan, top=12))


def test_exemplar_is_top_left_instance():
    paths = _symbol_at(200, 100) + _symbol_at(10, 10)
    scan = count.scan_symbols(_geom(paths))
    assert scan.clusters[0].exemplar_bbox[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# tag cross-check + report
# ---------------------------------------------------------------------------
def test_fixture_tag_counts_match_tag_shapes_only():
    words = [
        TextWord("WC-1", (0, 0, 10, 4)), TextWord("WC-1", (50, 0, 60, 4)),
        TextWord("LAV2", (0, 10, 10, 14)), TextWord("P-3", (0, 20, 10, 24)),
        TextWord("12-0", (0, 30, 10, 34)),   # pipe-length tag — not a fixture tag
        TextWord("2", (0, 40, 4, 44)),       # pipe size — no letters
        TextWord("TYP", (0, 50, 10, 54)),    # no digits
    ]
    tags = count.fixture_tag_counts(_geom([], words=words))
    assert tags == {"WC-1": 2, "LAV2": 1, "P-3": 1}


def test_scan_report_shows_counts_caps_and_tags():
    paths = _symbol_at(0, 0) + _symbol_at(100, 0)
    geom = _geom(paths, words=[TextWord("WC-1", (0, 0, 10, 4))])
    scan = count.scan_symbols(geom)
    report = count.build_scan_report(geom, scan)
    assert "S0" in report and "2" in report
    assert "WC-1 x1" in report
    assert "1.1x0.7 ft" in report            # 10x6 pt at 9 pt/ft


def test_scan_review_notes_surface_remainder_and_singletons():
    paths = (
        _symbol_at(0, 0) + _symbol_at(100, 0)                                    # S0 x2
        + [_path([("re", (0.0, 50.0, 4.0, 54.0))]), _path([("re", (40.0, 50.0, 44.0, 54.0))])]
        # an isolated extra copy of one arm: the part repeats (so it survives
        # stage 1) but the arrangement is one-off -> a singleton arrangement.
        + [_path(_translate(_ARM_A, 400, 300))]
    )
    scan = count.scan_symbols(_geom(paths))
    assert len(scan.singletons) == 1
    notes = count.scan_review_notes(scan, top=1)
    assert any("NOT REVIEWED: 1 smaller clusters" in n and "2 instances" in n for n in notes)
    assert any("SINGLETONS: 1 one-off arrangements" in n for n in notes)
