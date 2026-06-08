"""Hermetic tests for the M1 diagnostic logic (no PDF, no PyMuPDF).

Builds ``SheetGeometry`` by hand so the scale correlation and report assembly
are exercised on exact synthetic geometry.
"""
from __future__ import annotations

import pytest

from drawing_takeoff import diagnose
from drawing_takeoff.models import GeometryPath, SheetGeometry, SheetRef, TextWord


def _line(p0, p1, *, color, width, kind="stroke"):
    return GeometryPath(
        items=(("l", p0, p1),),
        stroke_color=color,
        fill_color=None,
        width=width,
        dashes="[] 0",
        closed=False,
        bbox=(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])),
        kind=kind,
    )


@pytest.fixture
def geom_with_one_pipe():
    pipe = _line((50, 100), (158, 100), color=(0.0, 0.0, 0.0), width=1.3)  # 108pt = 12ft
    background = _line((10, 10), (40, 10), color=(0.667, 0.667, 0.667), width=0.5)
    label = TextWord(text="12-0", bbox=(100, 105, 108, 111))  # centroid (104,108), 8pt off
    return SheetGeometry(
        ref=SheetRef("synthetic", 0),
        page_width_pt=216,
        page_height_pt=144,
        paths=[pipe, background],
        words=[label],
        scale_label='1/8" = 1\'-0"',
        points_per_foot=9.0,
    ), pipe


def test_scale_check_matches_pipe_label_to_run(geom_with_one_pipe):
    geom, pipe = geom_with_one_pipe
    matches, inferred = diagnose.scale_check(geom)
    assert len(matches) == 1
    m = matches[0]
    assert m.label == "12-0"
    assert m.measured_pt == pytest.approx(108.0)
    assert m.err_pct == pytest.approx(0.0, abs=0.01)
    assert m.style == pipe.style_key
    assert inferred == pipe.style_key


def test_build_report_passes_and_has_all_sections(geom_with_one_pipe):
    geom, _ = geom_with_one_pipe
    report = diagnose.build_report(geom)
    assert "(a) CLEANLINESS" in report
    assert "(b) INSTANCES" in report
    assert "(c) SCALE" in report
    assert "KNOWN-DIMENSION CHECK" in report
    assert "PASS" in report
    assert "1/8" in report  # the detected label echoes back


def test_scale_check_without_ppf_is_empty():
    geom = SheetGeometry(ref=SheetRef("x", 0), page_width_pt=10, page_height_pt=10)
    assert diagnose.scale_check(geom) == ([], None)


def test_length_helpers():
    assert diagnose._path_length(_line((0, 0), (108, 0), color=None, width=1)) == pytest.approx(108.0)
    # a straight-ish cubic approximates its chord
    straight = GeometryPath(
        items=(("c", (0, 0), (10, 0), (20, 0), (30, 0)),),
        stroke_color=None, fill_color=None, width=1, dashes="[] 0",
        closed=False, bbox=(0, 0, 30, 0), kind="stroke",
    )
    assert diagnose._path_length(straight) == pytest.approx(30.0, abs=1.0)
    # rectangle perimeter
    rect = GeometryPath(
        items=(("re", (0, 0, 10, 5)),),
        stroke_color=None, fill_color=None, width=1, dashes="[] 0",
        closed=True, bbox=(0, 0, 10, 5), kind="stroke",
    )
    assert diagnose._path_length(rect) == pytest.approx(30.0)
