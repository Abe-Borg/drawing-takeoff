"""Hermetic tests for the M4 pipeline assembly (no PDF, no PyMuPDF, no network).

``takeoff_for_sheet`` measures + labels a hand-built ``SheetGeometry`` with a
fake legend client, so the whole geometry->measure->legend->item path is
exercised without a backend.
"""
from __future__ import annotations

import json

import pytest

from drawing_takeoff import pipeline
from drawing_takeoff.models import GeometryPath, SheetGeometry, SheetRef, StyleKey, TakeoffItem
from tests.fixtures.fake_anthropic import FakeClient, FakeMessage, FakeTextBlock


def _line(p0, p1, *, color, width):
    return GeometryPath(
        items=(("l", p0, p1),),
        stroke_color=color,
        fill_color=None,
        width=width,
        dashes="[] 0",
        closed=False,
        bbox=(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])),
        kind="stroke",
    )


def _client(labels):
    return FakeClient(
        lambda kw: FakeMessage(
            content=[FakeTextBlock(text=json.dumps({"labels": labels}))]
        )
    )


@pytest.fixture
def geom():
    # black 1.3pt pipe (2 runs = 22 ft) ranks first; gray background second
    paths = [
        _line((0, 0), (108, 0), color=(0.0, 0.0, 0.0), width=1.3),
        _line((0, 20), (90, 20), color=(0.0, 0.0, 0.0), width=1.3),
        _line((0, 60), (100, 60), color=(0.667, 0.667, 0.667), width=0.5),
    ]
    return SheetGeometry(
        ref=SheetRef("FP.pdf", 0), page_width_pt=216, page_height_pt=144,
        paths=paths, words=[], scale_label='1/8" = 1\'-0"', points_per_foot=9.0,
    )


def test_takeoff_for_sheet_builds_named_items(geom):
    client = _client([
        {"style_id": "s0", "system": "Fire-protection sprinkler pipe", "measurable": True,
         "confidence": "high", "ambiguous": False, "reasoning": "heavy black, long runs"},
        {"style_id": "s1", "system": "Architectural background", "measurable": False,
         "confidence": "high", "ambiguous": False, "reasoning": "gray background"},
    ])
    items, diag = pipeline.takeoff_for_sheet(geom, client=client, discipline="fire protection")
    assert len(items) == 1  # only the measurable pipe style becomes an item
    it = items[0]
    assert it.system == "Fire-protection sprinkler pipe"
    assert it.unit == "LF"
    assert it.quantity == pytest.approx(22.0)  # (108 + 90) pt / 9
    assert it.sheet == "FP.pdf#p0"
    assert it.run_count == 2
    assert it.trusted
    assert any("ppf=9" in d for d in diag)


def test_uncertain_measured_style_is_flagged_not_dropped(geom):
    # gray labeled NOT measurable but AMBIGUOUS -> its footage must surface as
    # flagged, not silently disappear from the takeoff.
    client = _client([
        {"style_id": "s0", "system": "Pipe", "measurable": True, "confidence": "high",
         "ambiguous": False, "reasoning": ""},
        {"style_id": "s1", "system": "Possibly a branch", "measurable": False, "confidence": "low",
         "ambiguous": True, "reasoning": "gray, unsure vs background"},
    ])
    items, _ = pipeline.takeoff_for_sheet(geom, client=client)
    trusted = [i for i in items if i.trusted]
    flagged = [i for i in items if not i.trusted]
    assert any(i.system == "Pipe" for i in trusted)
    assert any(i.quantity > 0 and i.system == "Possibly a branch" for i in flagged)


def test_omitted_measured_style_is_flagged_not_dropped(geom):
    # the model returns only s0; s1 has real footage but is omitted -> flagged.
    client = _client([
        {"style_id": "s0", "system": "Pipe", "measurable": True, "confidence": "high",
         "ambiguous": False, "reasoning": ""},
    ])
    items, _ = pipeline.takeoff_for_sheet(geom, client=client)
    assert any(i.quantity > 0 and not i.trusted for i in items)  # gray surfaced, not dropped


def test_confident_background_is_excluded(geom):
    # gray labeled NOT measurable and NOT ambiguous -> correctly excluded (not flagged).
    client = _client([
        {"style_id": "s0", "system": "Pipe", "measurable": True, "confidence": "high",
         "ambiguous": False, "reasoning": ""},
        {"style_id": "s1", "system": "Architectural background", "measurable": False,
         "confidence": "high", "ambiguous": False, "reasoning": "gray background"},
    ])
    items, _ = pipeline.takeoff_for_sheet(geom, client=client)
    assert len(items) == 1 and items[0].system == "Pipe"


def test_takeoff_for_sheet_requires_scale(geom):
    geom.points_per_foot = None
    geom.scale_label = None
    with pytest.raises(ValueError):
        pipeline.takeoff_for_sheet(geom, client=_client([]))


def test_scale_label_override_wins(geom):
    # override to 1/4"=1'-0" (18 pt/ft) -> the 198pt of pipe now reads 11 ft
    client = _client([
        {"style_id": "s0", "system": "Pipe", "measurable": True, "confidence": "high",
         "ambiguous": False, "reasoning": ""},
    ])
    items, _ = pipeline.takeoff_for_sheet(geom, client=client, scale_label='1/4" = 1\'-0"')
    assert items[0].quantity == pytest.approx(11.0)
    assert items[0].scale_used == pytest.approx(18.0)


def _item(system, qty, sheet, *, conf="high", amb=False):
    return TakeoffItem(system, qty, "LF", sheet, StyleKey((0.0, 0.0, 0.0), 1.3, "[] 0"), 9.0,
                       confidence=conf, ambiguous=amb)


def test_aggregate_sums_trusted_across_sheets_only():
    items = [
        _item("Pipe", 100.0, "a#p0"),
        _item("Pipe", 50.0, "b#p0", conf="medium"),
        _item("Pipe", 999.0, "c#p0", conf="low", amb=True),  # flagged -> not counted
        _item("Duct", 30.0, "a#p0"),
    ]
    assert pipeline._aggregate(items) == {"Pipe": 150.0, "Duct": 30.0}
