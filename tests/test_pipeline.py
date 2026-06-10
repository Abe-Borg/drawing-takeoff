"""Hermetic tests for the M4 pipeline assembly (no PDF, no PyMuPDF, no network).

``takeoff_for_sheet`` measures + labels a hand-built ``SheetGeometry`` with a
fake legend client, so the whole geometry->measure->legend->item path is
exercised without a backend.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from drawing_takeoff import pipeline
from drawing_takeoff.models import (
    GeometryPath,
    SheetGeometry,
    SheetRef,
    StyleKey,
    SystemSizeResult,
    SystemSizeSheet,
    TakeoffItem,
)
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


def test_takeoff_for_sheet_passes_legend_block_through(geom):
    captured = {}

    def responder(kw):
        captured.update(kw)
        return FakeMessage(content=[FakeTextBlock(text=json.dumps({"labels": [
            {"style_id": "s0", "system": "Pipe", "measurable": True, "confidence": "high",
             "ambiguous": False, "reasoning": ""}]}))])

    block = {"type": "document", "source": {"type": "file", "file_id": "file_xyz"}}
    pipeline.takeoff_for_sheet(geom, client=FakeClient(responder), legend_block=block)
    assert block in captured["messages"][0]["content"]


def test_extract_takeoff_uploads_legend_once_and_cleans_up(tmp_path):
    # PyMuPDF-gated: two tiny synthetic sheets, one shared legend. The legend
    # must upload ONCE, ride every per-sheet request as a file reference (not
    # inline bytes), and be deleted when the run finishes.
    fitz = pytest.importorskip("fitz")

    paths = []
    for n in range(2):
        p = tmp_path / f"sheet{n}.pdf"
        doc = fitz.open()
        page = doc.new_page(width=216, height=144)
        shape = page.new_shape()
        shape.draw_line(fitz.Point(20, 50), fitz.Point(110, 50))
        shape.finish(color=(0, 0, 0), width=1.3)
        shape.commit()
        page.insert_text(fitz.Point(20, 20), '1/8" = 1\'-0"', fontsize=8)
        doc.save(p)
        doc.close()
        paths.append(str(p))

    contents = []

    def responder(kw):
        contents.append(kw["messages"][0]["content"])
        return FakeMessage(content=[FakeTextBlock(text=json.dumps({"labels": [
            {"style_id": "s0", "system": "Pipe", "measurable": True, "confidence": "high",
             "ambiguous": False, "reasoning": ""}]}))])

    client = FakeClient(responder, with_files=True)
    result = pipeline.extract_takeoff(paths, client=client, legend_pdf=b"%PDF-legend")
    assert not result.errors
    assert len(client.beta.files.uploads) == 1  # uploaded once, not per sheet
    file_ids = set()
    for content in contents:
        file_blocks = [b for b in content
                       if isinstance(b, dict) and (b.get("source") or {}).get("type") == "file"]
        assert len(file_blocks) == 1  # reference attached, no inline PDF copy
        file_ids.add(file_blocks[0]["source"]["file_id"])
    assert len(file_ids) == 1
    assert client.beta.files.deleted == sorted(file_ids)  # cleaned up after the run

    # A single sheet skips the upload round trip and inlines the bytes.
    single = FakeClient(responder, with_files=True)
    pipeline.extract_takeoff(paths[:1], client=single, legend_pdf=b"%PDF-legend")
    assert single.beta.files.uploads == []


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


# ---------------------------------------------------------------------------
# System × Size pipeline (M5–M8): networks -> system -> size, aggregated
# ---------------------------------------------------------------------------
def _size_client():
    """Fake client for the System×Size path: answers the style pass (s0 = pipe)
    and the network pass (every network id the facts mention -> trusted pipe),
    branching on which schema the request enforces."""

    def responder(kw):
        props = kw["output_config"]["format"]["schema"]["properties"]["labels"]["items"]["properties"]
        if "style_id" in props:  # M3 style pass
            return FakeMessage(content=[FakeTextBlock(text=json.dumps({"labels": [
                {"style_id": "s0", "system": "FP sprinkler", "measurable": True,
                 "confidence": "high", "ambiguous": False, "reasoning": "heavy black"},
            ]}))])
        # M7 network pass: read the ids straight from the per-network facts text.
        text = " ".join(
            b.get("text", "") for b in kw["messages"][0]["content"]
            if isinstance(b, dict) and b.get("type") == "text"
        )
        ids = list(dict.fromkeys(re.findall(r"\bN\d+\b", text)))
        return FakeMessage(content=[FakeTextBlock(text=json.dumps({"labels": [
            {"network_id": nid, "system": "FP sprinkler", "is_pipe": True,
             "confidence": "high", "ambiguous": False, "reasoning": "main"} for nid in ids
        ]}))])

    return FakeClient(responder)


def test_absorb_sheet_sums_tags_and_prefixes():
    # Pure aggregation: sum the System×Size tables, tag detail rows with their
    # sheet, prefix review notes with the sheet id — no PyMuPDF, no client.
    result = SystemSizeResult()
    pipeline._absorb_sheet(result, SystemSizeSheet(
        sheet="a#p0", source="a", page_index=0, networks=[1],
        tables={"by_system_size": {("FP", '2"'): 10.0, ("FP", "unsized"): 5.0},
                "detail": [{"network": "N0"}], "review": ["check N0"]},
    ))
    pipeline._absorb_sheet(result, SystemSizeSheet(
        sheet="b#p0", source="b", page_index=0, networks=[1],
        tables={"by_system_size": {("FP", '2"'): 4.0}, "detail": [{"network": "N0"}], "review": []},
    ))
    assert result.by_system_size[("FP", '2"')] == pytest.approx(14.0)
    assert result.by_system_size[("FP", "unsized")] == pytest.approx(5.0)
    assert {r["sheet"] for r in result.detail} == {"a#p0", "b#p0"}
    assert result.review == ["a#p0: check N0"]
    assert result.per_system_totals == {"FP": 19.0}
    assert len(result.sheets) == 2


def test_extract_system_size_takeoff_aggregates_across_sheets(tmp_path):
    # End-to-end over two tiny real sheets (exact geometry + real network render),
    # with the labels scripted: the per-sheet System×Size tables must sum into one
    # set-wide total and every detail row must carry its sheet.
    fitz = pytest.importorskip("fitz")

    def make(path):
        doc = fitz.open()
        page = doc.new_page(width=216, height=144)
        shape = page.new_shape()
        shape.draw_line(fitz.Point(20, 50), fitz.Point(150, 50))  # one 130 pt run -> one network
        shape.finish(color=(0, 0, 0), width=1.3)
        shape.commit()
        page.insert_text(fitz.Point(20, 20), '1/8" = 1\'-0"', fontsize=8)
        doc.save(path)
        doc.close()

    paths = []
    for n in range(2):
        p = tmp_path / f"s{n}.pdf"
        make(str(p))
        paths.append(str(p))

    result = pipeline.extract_system_size_takeoff(paths, client=_size_client())
    assert result.sheet_count == 2 and not result.errors
    assert ("FP sprinkler", "unsized") in result.by_system_size
    per_sheet_lf = 130 / 9.0  # (150 - 20) pt at 9 pt/ft
    assert result.per_system_totals["FP sprinkler"] == pytest.approx(2 * per_sheet_lf, rel=0.05)
    assert len(result.sheets) == 2
    assert result.detail and all("sheet" in row for row in result.detail)


def test_extract_system_size_takeoff_records_bad_sheet_without_sinking_run(tmp_path):
    # One unreadable PDF is captured in errors; the run still completes.
    pytest.importorskip("fitz")
    result = pipeline.extract_system_size_takeoff(
        [str(tmp_path / "missing.pdf")], client=_size_client()
    )
    assert result.sheet_count == 0
    assert any("could not read PDF" in e for e in result.errors)


def test_write_system_size_export_writes_workbook_and_one_markup_per_sheet(tmp_path, monkeypatch):
    # The Excel workbook always lands; a marked-up PDF is written per sheet that
    # has networks (the markup render itself is stubbed — geometry is covered above).
    pytest.importorskip("fitz")
    import drawing_takeoff.geometry as _geom

    def fake_markup(src, page_index, networks, out):
        Path(out).write_bytes(b"%PDF-markup")

    monkeypatch.setattr(_geom, "write_marked_up_pdf", fake_markup)

    result = SystemSizeResult(sheet_count=1)
    pipeline._absorb_sheet(result, SystemSizeSheet(
        sheet="FP.pdf#p0", source="FP.pdf", page_index=0, networks=[object()],
        tables={"by_system_size": {("FP", '2"'): 12.0}, "detail": [], "review": ["note"]},
    ))
    folder = pipeline.write_system_size_export(result, tmp_path)
    assert (folder / "takeoff.xlsx").exists()
    markups = sorted(p.name for p in folder.glob("*_markup.pdf"))
    assert markups == ["FP_p0_markup.pdf"]
