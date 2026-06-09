"""Hermetic tests for measurement primitives (no PDF, no PyMuPDF).

The stitcher is the milestone's crux, so its behavior is pinned explicitly:
collinear fragments merge into one run; a bend or a tee starts a new run;
exact duplicates don't double-count.
"""
from __future__ import annotations

import pytest

from drawing_takeoff import measure
from drawing_takeoff.models import GeometryPath, SheetGeometry, SheetRef, StyleKey, TextWord

BLACK13 = dict(stroke_color=(0.0, 0.0, 0.0), width=1.3, dashes="[] 0")


def _line(p0, p1, **style):
    s = {**BLACK13, **style}
    return GeometryPath(
        items=(("l", p0, p1),),
        stroke_color=s["stroke_color"],
        fill_color=None,
        width=s["width"],
        dashes=s["dashes"],
        closed=False,
        bbox=(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])),
        kind="stroke",
    )


# ---- length primitives ----------------------------------------------------
def test_polyline_length():
    assert measure.polyline_length_pt([(0, 0), (3, 0), (3, 4)]) == pytest.approx(7.0)
    assert measure.polyline_length_pt([(0, 0)]) == 0.0


def test_bezier_length_of_straight_control_points():
    # control points on a straight line -> length ~= chord
    assert measure.bezier_length_pt((0, 0), (10, 0), (20, 0), (30, 0)) == pytest.approx(30.0, abs=1e-6)


def test_group_by_style():
    a = _line((0, 0), (10, 0))
    b = _line((0, 5), (10, 5))
    c = _line((0, 0), (10, 0), width=0.5)
    grouped = measure.group_by_style([a, b, c])
    assert len(grouped) == 2
    assert len(grouped[a.style_key]) == 2
    assert len(grouped[c.style_key]) == 1


# ---- stitching: the crux --------------------------------------------------
def test_collinear_fragments_become_one_run():
    # three collinear pieces of a 108pt line, as CAD would fragment it
    frags = [_line((0, 0), (36, 0)), _line((36, 0), (72, 0)), _line((72, 0), (108, 0))]
    runs = measure.stitch_runs(frags, ppf=9.0)
    assert len(runs) == 1
    assert runs[0].length_pt == pytest.approx(108.0)
    assert runs[0].length_ft == pytest.approx(12.0)
    assert runs[0].segment_count == 3
    assert runs[0].polyline[0] == (0, 0) and runs[0].polyline[-1] == (108, 0)


def test_bend_splits_into_two_runs():
    # an elbow: horizontal then vertical -> not collinear -> two runs
    parts = [_line((0, 0), (108, 0)), _line((108, 0), (108, 50))]
    runs = measure.stitch_runs(parts, ppf=9.0)
    assert len(runs) == 2
    lengths = sorted(r.length_pt for r in runs)
    assert lengths == pytest.approx([50.0, 108.0])


def test_tee_keeps_through_run_and_splits_branch():
    # collinear through-run (200pt) plus a branch (50pt) at the midpoint
    parts = [
        _line((0, 0), (100, 0)),
        _line((100, 0), (200, 0)),   # collinear with the first -> one 200pt run
        _line((100, 0), (100, 50)),  # branch -> its own run
    ]
    runs = measure.stitch_runs(parts, ppf=9.0)
    assert len(runs) == 2
    lengths = sorted(r.length_pt for r in runs)
    assert lengths == pytest.approx([50.0, 200.0])


def test_duplicate_segments_are_not_double_counted():
    parts = [_line((0, 0), (108, 0)), _line((0, 0), (108, 0))]  # drawn twice
    runs = measure.stitch_runs(parts, ppf=9.0)
    assert len(runs) == 1
    assert runs[0].length_pt == pytest.approx(108.0)  # not 216


def test_endpoint_tolerance_honored_across_snap_boundary():
    # Shared endpoint jittered across a grid-rounding boundary: 40.24 vs 40.26
    # are 0.02pt apart but a naive snap buckets them apart. They must still merge.
    a = _line((10, 0), (40.24, 0))
    b = _line((40.26, 0), (70, 0))
    runs = measure.stitch_runs([a, b], tol=0.5)
    assert len(runs) == 1
    assert runs[0].length_pt == pytest.approx(59.98, abs=0.01)


def test_near_but_not_collinear_does_not_merge():
    # 3 degrees apart, sharing an endpoint -> beyond the 2deg tol -> two runs
    import math
    L = 100.0
    a = _line((0, 0), (L, 0))
    dx, dy = L * math.cos(math.radians(3)), L * math.sin(math.radians(3))
    b = _line((L, 0), (L + dx, dy))
    runs = measure.stitch_runs([a, b])
    assert len(runs) == 2


# ---- per-style totals + association ---------------------------------------
def _sheet(paths, ppf=9.0):
    return SheetGeometry(
        ref=SheetRef("synthetic", 0), page_width_pt=500, page_height_pt=500,
        paths=paths, points_per_foot=ppf,
    )


def test_linear_feet_by_style():
    pipe = [_line((0, 0), (108, 0)), _line((0, 20), (90, 20))]          # 12ft + 10ft
    other = [_line((0, 40), (45, 40), width=0.5)]                        # 5ft, other style
    feet = measure.linear_feet_by_style(_sheet(pipe + other))
    assert feet[pipe[0].style_key] == pytest.approx(22.0)
    assert feet[other[0].style_key] == pytest.approx(5.0)


def test_linear_feet_requires_scale():
    with pytest.raises(ValueError):
        measure.linear_feet_by_style(_sheet([_line((0, 0), (9, 0))], ppf=None))


def test_nearest_run_associates_label_location():
    runs = measure.stitch_runs([_line((0, 100), (108, 100))], ppf=9.0)
    near = measure.nearest_run((54, 108), runs, max_dist_pt=30)
    assert near is not None and near.length_pt == pytest.approx(108.0)
    assert measure.nearest_run((54, 400), runs, max_dist_pt=30) is None


# ---- M2 report layer ------------------------------------------------------
def test_length_tag_total_sums_callouts_only():
    geom = _sheet([_line((0, 0), (108, 0))])
    geom.words = [
        TextWord("12-0", (100, 105, 108, 111)),   # 12.0 ft
        TextWord("10-0", (100, 125, 108, 131)),   # 10.0 ft
        TextWord("N-145", (0, 0, 10, 5)),         # node id, not a length
    ]
    n, ft = measure.length_tag_total(geom)
    assert (n, ft) == (2, pytest.approx(22.0))


def test_heaviest_dark_style_picks_thickest_black():
    thin = StyleKey((0.0, 0.0, 0.0), 0.24, "[] 0")
    thick = StyleKey((0.0, 0.0, 0.0), 1.3, "[] 0")
    gray = StyleKey((0.67, 0.67, 0.67), 0.5, "[] 0")
    assert measure.heaviest_dark_style([thin, thick, gray]) == thick
    assert measure.heaviest_dark_style([gray]) is None  # nothing dark


def test_build_measure_report_cross_checks_against_tags():
    pipe = [_line((0, 0), (108, 0)), _line((0, 20), (90, 20))]  # 12 + 10 = 22 LF
    geom = _sheet(pipe)
    geom.scale_label = '1/8" = 1\'-0"'
    geom.words = [TextWord("12-0", (50, 5, 60, 11)), TextWord("10-0", (40, 25, 50, 31))]
    report = measure.build_measure_report(geom)
    assert "TAKEOFF lineweight" in report
    assert "CROSS-CHECK" in report
    assert "22.0 LF" in report
    assert "+0.00%" in report  # measured total == callout total


def test_build_measure_report_without_scale():
    geom = _sheet([_line((0, 0), (9, 0))], ppf=None)
    assert "no scale" in measure.build_measure_report(geom)


# ---- border / matchline exclusion -----------------------------------------
def test_is_border_run_spans_and_hugs_an_edge():
    # page 3024x2160. The border spans the sheet AND hugs an edge.
    edge = measure.stitch_runs([_line((0, 2160), (3024, 2160))], ppf=9.0)[0]  # bottom edge
    assert measure.is_border_run(edge, 3024, 2160) is True
    # a 62 ft riser in the interior (not full-span) is kept
    riser = measure.stitch_runs([_line((900, 600), (900, 1162))], ppf=9.0)[0]
    assert measure.is_border_run(riser, 3024, 2160) is False


def test_interior_full_width_main_is_kept():
    # a long main crossing 86% of the width but in the plan interior (not hugging
    # the top/bottom edge) must NOT be classified as a border (Codex PR#7).
    main = measure.stitch_runs([_line((100, 1000), (2700, 1000))], ppf=9.0)[0]
    assert measure.is_border_run(main, 3024, 2160) is False
    # a diagonal spanning the sheet is also kept (not axis-aligned at an edge)
    diag = measure.stitch_runs([_line((100, 100), (2900, 2000))], ppf=9.0)[0]
    assert measure.is_border_run(diag, 3024, 2160) is False


def test_linear_feet_excludes_border_by_default():
    # a full-width border (its own pen) + a 12 ft pipe run, on a real-size sheet
    border = _line((0, 0), (3024, 0), stroke_color=(0.0, 0.0, 0.0), width=0.86)
    pipe = _line((10, 100), (118, 100))  # 108 pt = 12 ft, black 1.3 (3.6% of width)
    geom = SheetGeometry(
        ref=SheetRef("x", 0), page_width_pt=3024, page_height_pt=2160,
        paths=[border, pipe], points_per_foot=9.0,
    )
    feet = measure.linear_feet_by_style(geom, 9.0)          # exclude_border defaults True
    assert max(feet.values()) < 150                         # the 336 ft border total is gone
    assert any(abs(v - 12.0) < 0.5 for v in feet.values())  # the pipe run is kept

    raw = measure.linear_feet_by_style(geom, 9.0, exclude_border=False)
    assert max(raw.values()) > 300                          # raw still includes the 336 ft border


# ---- M5 network connectivity ----------------------------------------------
def _runs(*segments, ppf=9.0):
    """One stitched Run per (p0, p1) segment — the input shape ``networks`` takes."""
    out = []
    for p0, p1 in segments:
        out.extend(measure.stitch_runs([_line(p0, p1)], ppf=ppf))
    return out


def test_networks_join_a_branch_teeing_into_a_main():
    # a branch whose endpoint lands on the MIDDLE of the main must join it: this
    # is the endpoint-to-SEGMENT case endpoint-only connectivity misses.
    main = measure.stitch_runs([_line((0, 0), (200, 0))], ppf=9.0)
    branch = measure.stitch_runs([_line((100, 0), (100, 60))], ppf=9.0)
    nets = measure.networks(main + branch, ppf=9.0)
    assert len(nets) == 1
    assert nets[0].run_count == 2
    assert nets[0].length_ft == pytest.approx((200 + 60) / 9.0)
    assert nets[0].id == "N0"


def test_networks_separate_disjoint_systems():
    far = _runs(((0, 0), (100, 0)), ((0, 500), (100, 500)))  # 500 pt apart, well beyond tol
    nets = measure.networks(far, ppf=9.0)
    assert len(nets) == 2
    assert all(nw.run_count == 1 for nw in nets)


def test_networks_bridge_a_fitting_gap_within_tolerance():
    # two collinear runs with a 3 pt break (pipe broken at a fitting): 0.5 ft tol
    # (= 4.5 pt at 9 pt/ft) bridges it; a 30 pt gap stays two networks.
    a = measure.stitch_runs([_line((0, 0), (100, 0))], ppf=9.0)
    near = measure.stitch_runs([_line((103, 0), (200, 0))], ppf=9.0)
    assert len(measure.networks(a + near, ppf=9.0, tol_ft=0.5)) == 1
    far = measure.stitch_runs([_line((130, 0), (200, 0))], ppf=9.0)
    assert len(measure.networks(a + far, ppf=9.0, tol_ft=0.5)) == 2


def test_networks_do_not_merge_a_crossover():
    # a horizontal and a vertical run cross at (50, 0) but NEITHER has an endpoint
    # there -> not connected. This is the crossover-safety the ~0.5 ft tol relies on.
    horiz = measure.stitch_runs([_line((0, 0), (100, 0))], ppf=9.0)
    vert = measure.stitch_runs([_line((50, -50), (50, 50))], ppf=9.0)
    assert len(measure.networks(horiz + vert, ppf=9.0, tol_ft=0.5)) == 2


def test_networks_sorted_largest_first():
    nets = measure.networks(_runs(((0, 0), (50, 0)), ((0, 100), (300, 100))), ppf=9.0)
    assert [nw.id for nw in nets] == ["N0", "N1"]
    assert nets[0].length_pt > nets[1].length_pt
    assert nets[0].style_keys  # a network exposes the styles it spans


def test_build_networks_report_reports_a_connected_system():
    # main + branch in the pipe pen on a real-size, scaled sheet -> one network
    paths = [_line((100, 100), (1000, 100)), _line((500, 100), (500, 400))]
    geom = SheetGeometry(
        ref=SheetRef("x", 0), page_width_pt=3024, page_height_pt=2160,
        paths=paths, points_per_foot=9.0, scale_label='1/8" = 1\'-0"',
    )
    report = measure.build_networks_report(geom)
    assert "M5 networks" in report
    assert "1 network(s)" in report
    assert "N0" in report


def test_build_networks_report_without_scale():
    geom = _sheet([_line((0, 0), (9, 0))], ppf=None)
    assert "no scale" in measure.build_networks_report(geom)


def test_networks_connect_along_a_diagonal_main():
    # a 45deg main + a branch whose endpoint lands on it -> one network. Exercises
    # the diagonal grid-traversal registration (a bbox fill would also work but is
    # the pathological case the walk replaces).
    main = measure.stitch_runs([_line((0, 0), (200, 200))], ppf=9.0)
    branch = measure.stitch_runs([_line((100, 100), (100, 160))], ppf=9.0)
    nets = measure.networks(main + branch, ppf=9.0)
    assert len(nets) == 1
    assert nets[0].run_count == 2


def test_build_networks_report_skips_empty_border_style():
    # a heavy black sheet border (its own pen, fully border-excluded) is *thicker*
    # than the pipe pen; the candidate pick must skip the now-empty border style
    # and still find the pipe, not report "no candidate".
    border = _line((0, 0), (3024, 0), stroke_color=(0.0, 0.0, 0.0), width=2.0)
    pipe = [_line((100, 100), (1000, 100)), _line((500, 100), (500, 400))]  # black 1.3
    geom = SheetGeometry(
        ref=SheetRef("x", 0), page_width_pt=3024, page_height_pt=2160,
        paths=[border, *pipe], points_per_foot=9.0, scale_label='1/8" = 1\'-0"',
    )
    report = measure.build_networks_report(geom)
    assert "no candidate" not in report
    assert "1 network(s)" in report
