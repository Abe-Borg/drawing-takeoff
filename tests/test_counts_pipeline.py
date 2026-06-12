"""Hermetic tests for the counts takeoff: M10 naming (fake client), M11
assembly (pure), the pipeline orchestration over tiny synthetic PDFs
(PyMuPDF-gated), and the counts workbook."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from drawing_takeoff import count, export, legend, pipeline
from drawing_takeoff.models import (
    CountsResult,
    CountsSheet,
    SheetGeometry,
    SheetRef,
    SymbolCluster,
    SystemLabel,
    TextWord,
)
from tests.fixtures.fake_anthropic import FakeClient, FakeMessage, FakeTextBlock


def _cluster(cid, n, *, variants=1):
    boxes = tuple((i * 50.0, 0.0, i * 50.0 + 9.0, 9.0) for i in range(n))
    return SymbolCluster(
        id=cid, signature_hash="cafe00", instance_bboxes=boxes,
        paths_per_instance=2, width_pt=9.0, height_pt=9.0, variants=variants,
    )


def _geom(words=()):
    return SheetGeometry(
        ref=SheetRef("A.pdf", 0), page_width_pt=612, page_height_pt=396,
        paths=[], words=list(words), scale_label='1/8" = 1\'-0"', points_per_foot=9.0,
    )


def _label(component, *, countable=True, conf="high", amb=False, reasoning=""):
    return SystemLabel(system=component, measurable=countable, confidence=conf,
                       ambiguous=amb, reasoning=reasoning)


# ---------------------------------------------------------------------------
# M10: label_symbols / second_look_symbols
# ---------------------------------------------------------------------------
def test_label_symbols_maps_components_and_flags_omitted():
    captured = {}

    def responder(kw):
        captured.update(kw)
        return FakeMessage(content=[FakeTextBlock(text=json.dumps({"labels": [
            {"symbol_id": "S0", "component": "Sprinkler head", "countable": True,
             "confidence": "high", "ambiguous": False, "reasoning": "donut on pipe"},
        ]}))])

    geom = _geom()
    clusters = [_cluster("S0", 3), _cluster("S1", 2)]
    labels = legend.label_symbols(
        geom, clusters, client=FakeClient(responder), crops={"S0": b"png0", "S1": b"png1"},
    )
    assert labels["S0"].system == "Sprinkler head" and labels["S0"].measurable
    assert labels["S0"].trusted
    # S1 omitted by the model -> flagged default, never silently trusted.
    assert labels["S1"].ambiguous and not labels["S1"].measurable

    text = " ".join(b.get("text", "") for b in captured["messages"][0]["content"]
                    if isinstance(b, dict) and b.get("type") == "text")
    assert "S0: 3 instances" in text and "S1: 2 instances" in text
    images = [b for b in captured["messages"][0]["content"]
              if isinstance(b, dict) and b.get("type") == "image"]
    assert len(images) == 2  # one exemplar crop per cluster
    schema = captured["output_config"]["format"]["schema"]
    props = schema["properties"]["labels"]["items"]["properties"]
    assert "symbol_id" in props and "countable" in props
    # The contract: no numeric fields the model could smuggle a count through.
    assert all(p.get("type") != "number" for p in props.values())


def test_label_symbols_attaches_advisory_legend():
    captured = {}

    def responder(kw):
        captured.update(kw)
        return FakeMessage(content=[FakeTextBlock(text=json.dumps({"labels": []}))])

    legend.label_symbols(_geom(), [_cluster("S0", 2)], client=FakeClient(responder),
                         crops={"S0": b"p"}, legend_pdf=b"%PDF-legend")
    docs = [b for b in captured["messages"][0]["content"]
            if isinstance(b, dict) and b.get("type") == "document"]
    assert len(docs) == 1
    text = " ".join(b.get("text", "") for b in captured["messages"][0]["content"]
                    if isinstance(b, dict) and b.get("type") == "text")
    assert "advisory" in text


def test_second_look_symbols_prefixes_and_only_returns_given_ids():
    def responder(kw):
        return FakeMessage(content=[FakeTextBlock(text=json.dumps({"labels": [
            {"symbol_id": "S2", "component": "Floor drain", "countable": True,
             "confidence": "high", "ambiguous": False, "reasoning": "round grate at low point"},
        ]}))])

    geom = _geom()
    flagged = [_cluster("S2", 4)]
    first = {"S2": _label("round thing?", countable=True, conf="low", amb=True)}
    out = legend.second_look_symbols(
        geom, flagged, first, {"S2": [b"a", b"b"]}, client=FakeClient(responder),
    )
    assert set(out) == {"S2"}
    assert out["S2"].trusted and out["S2"].reasoning.startswith("second look:")


def test_second_look_symbols_skips_clusters_without_crops():
    out = legend.second_look_symbols(_geom(), [_cluster("S0", 2)], {}, {}, client=None)
    assert out == {}


# ---------------------------------------------------------------------------
# M11: counts assembly (pure)
# ---------------------------------------------------------------------------
def test_counts_takeoff_trust_gate_and_name_reunification():
    clusters = [_cluster("S0", 47), _cluster("S1", 12), _cluster("S2", 5), _cluster("S3", 9)]
    labels = {
        "S0": _label("Sprinkler head"),
        "S1": _label("Sprinkler head", conf="medium"),     # sibling cluster, same name
        "S2": _label("Size label", countable=False),       # annotation: excluded by name
        "S3": _label("Maybe a drain", conf="low", amb=True),
    }
    by, review = count.counts_takeoff(clusters, labels)
    assert by == {"Sprinkler head": 59}                    # 47 + 12 reunified by name
    assert any("not countable (Size label)" in r for r in review)
    assert any("Maybe a drain" in r and "9 EA" in r for r in review)


def test_counts_takeoff_confirm_note_for_multi_variant_cluster():
    by, review = count.counts_takeoff([_cluster("S0", 6, variants=4)], {"S0": _label("Head")})
    assert by == {"Head": 6}
    assert any("CONFIRM" in r and "4 drawn variants" in r for r in review)


def test_count_tables_and_report_shapes():
    geom = _geom(words=[TextWord("FD-1", (0, 0, 10, 4)), TextWord("FD-1", (50, 0, 60, 4))])
    clusters = [_cluster("S0", 3), _cluster("S1", 2)]
    labels = {"S0": _label("Floor drain"), "S1": _label("Hatch", countable=False)}
    tables = count.count_tables(clusters, labels, geom)
    assert tables["by_component"] == {"Floor drain": 3}
    assert [r["cluster"] for r in tables["detail"]] == ["S0", "S1"]
    assert tables["detail"][0]["counted"] and not tables["detail"][1]["counted"]
    assert tables["detail"][0]["size"] == "1.0x1.0 ft"

    report = count.build_counts_report(geom, clusters, labels)
    assert "Floor drain: 3 EA" in report and "[S0 x3]" in report
    assert "not countable (Hatch)" in report
    assert "FD-1 x2" in report   # the advisory tag cross-check


def test_absorb_counts_sheet_sums_and_prefixes():
    result = CountsResult()
    pipeline._absorb_counts_sheet(result, CountsSheet(
        sheet="a#p0", source="a", page_index=0, clusters=[1],
        tables={"by_component": {"WC": 4}, "detail": [{"cluster": "S0"}], "review": ["check S0"]},
    ))
    pipeline._absorb_counts_sheet(result, CountsSheet(
        sheet="b#p0", source="b", page_index=0, clusters=[1],
        tables={"by_component": {"WC": 3, "Lavatory": 2}, "detail": [{"cluster": "S0"}], "review": []},
    ))
    assert result.by_component == {"WC": 7, "Lavatory": 2}
    assert result.per_component_totals == {"WC": 7, "Lavatory": 2}
    assert {r["sheet"] for r in result.detail} == {"a#p0", "b#p0"}
    assert result.review == ["a#p0: check S0"]


# ---------------------------------------------------------------------------
# pipeline end to end over tiny synthetic sheets (PyMuPDF-gated)
# ---------------------------------------------------------------------------
def _make_symbol_sheet(path, n_symbols=3):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=216, height=144)
    shape = page.new_shape()
    for i in range(n_symbols):
        x = 30 + i * 60
        # finish() per primitive: each becomes its own drawing, so a symbol is
        # two loose paths — the flattened-block reality the engine assembles.
        shape.draw_rect(fitz.Rect(x, 50, x + 9, 59))
        shape.finish(color=(0, 0, 0), width=0.7)
        shape.draw_line(fitz.Point(x + 1, 54.5), fitz.Point(x + 8, 54.5))
        shape.finish(color=(0, 0, 0), width=0.7)
    shape.commit()
    page.insert_text(fitz.Point(20, 20), '1/8" = 1\'-0"', fontsize=8)
    doc.save(path)
    doc.close()


def _counts_client(component="Widget", *, first_pass_flags=False):
    """Answers the symbol schema with one label per id found in the facts text.
    With ``first_pass_flags``, the first call returns ambiguous labels (forcing
    the second look) and later calls return confident ones."""
    calls = {"n": 0}

    def responder(kw):
        calls["n"] += 1
        text = " ".join(b.get("text", "") for b in kw["messages"][0]["content"]
                        if isinstance(b, dict) and b.get("type") == "text")
        ids = list(dict.fromkeys(re.findall(r"\bS\d+\b", text)))
        flag = first_pass_flags and calls["n"] == 1
        return FakeMessage(content=[FakeTextBlock(text=json.dumps({"labels": [
            {"symbol_id": sid, "component": "blob?" if flag else component,
             "countable": True, "confidence": "low" if flag else "high",
             "ambiguous": flag, "reasoning": "unsure" if flag else "clear shape"}
            for sid in ids
        ]}))])

    return FakeClient(responder), calls


def test_extract_count_takeoff_end_to_end(tmp_path):
    pytest.importorskip("fitz")
    paths = []
    for n in range(2):
        p = tmp_path / f"s{n}.pdf"
        _make_symbol_sheet(str(p))
        paths.append(str(p))

    client, _ = _counts_client()
    logs: list[str] = []
    result = pipeline.extract_count_takeoff(paths, client=client, log=logs.append)
    assert not result.errors and result.sheet_count == 2
    assert result.by_component == {"Widget": 6}            # 3 per sheet, summed
    assert all("sheet" in row for row in result.detail)
    assert any("naming with Claude" in m for m in logs)
    assert all(s.report for s in result.sheets)


def test_counts_for_sheet_second_look_recovers_flagged(tmp_path):
    pytest.importorskip("fitz")
    p = tmp_path / "s.pdf"
    _make_symbol_sheet(str(p))
    from drawing_takeoff.geometry import extract_pdf_geometry

    geom = extract_pdf_geometry(str(p))[0]
    client, calls = _counts_client(component="Floor drain", first_pass_flags=True)
    sheet = pipeline.counts_for_sheet(geom, client=client)
    assert calls["n"] == 2                                 # first pass + second look
    assert sheet.tables["by_component"] == {"Floor drain": 3}
    lab = sheet.labels["S0"]
    assert lab.trusted and lab.reasoning.startswith("second look:")


def test_counts_for_sheet_without_scale_still_counts(tmp_path):
    # Counts are scale-free: a sheet with no scale label must not raise.
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "noscale.pdf"
    doc = fitz.open()
    page = doc.new_page(width=216, height=144)
    shape = page.new_shape()
    for x in (30, 90):
        shape.draw_rect(fitz.Rect(x, 50, x + 9, 59))
        shape.finish(color=(0, 0, 0), width=0.7)
    shape.commit()
    doc.save(str(p))
    doc.close()
    from drawing_takeoff.geometry import extract_pdf_geometry

    geom = extract_pdf_geometry(str(p))[0]
    client, _ = _counts_client()
    sheet = pipeline.counts_for_sheet(geom, client=client)
    assert sheet.tables["by_component"] == {"Widget": 2}


def test_extract_count_takeoff_records_bad_sheet_without_sinking_run(tmp_path):
    pytest.importorskip("fitz")
    client, _ = _counts_client()
    result = pipeline.extract_count_takeoff([str(tmp_path / "missing.pdf")], client=client)
    assert result.sheet_count == 0
    assert any("could not read PDF" in e for e in result.errors)


def test_write_count_export_writes_workbook_and_markups(tmp_path, monkeypatch):
    pytest.importorskip("fitz")
    import drawing_takeoff.geometry as _geom_mod

    written = []

    def fake_markup(src, page_index, clusters, out, labels=None):
        written.append(out)
        Path(out).write_bytes(b"%PDF-markup")

    monkeypatch.setattr(_geom_mod, "write_count_markup_pdf", fake_markup)

    result = CountsResult(sheet_count=1)
    pipeline._absorb_counts_sheet(result, CountsSheet(
        sheet="A.pdf#p0", source="A.pdf", page_index=0, clusters=[_cluster("S0", 3)],
        labels={"S0": _label("WC")},
        tables={"by_component": {"WC": 3}, "detail": [], "review": ["note"]},
    ))
    folder = pipeline.write_count_export(result, tmp_path)
    assert (folder / "counts.xlsx").exists()
    assert [Path(w).name for w in written] == ["A_p0_counts_markup.pdf"]


# ---------------------------------------------------------------------------
# counts workbook
# ---------------------------------------------------------------------------
def test_build_counts_workbook_cells():
    tables = {
        "by_component": {"Sprinkler head": 59, "Floor drain": 3},
        "detail": [
            {"sheet": "a#p0", "cluster": "S0", "component": "Sprinkler head", "countable": True,
             "counted": True, "confidence": "high", "ambiguous": False, "count": 47,
             "size": "1.0x1.0 ft", "variants": 3, "reasoning": "donut on pipe"},
        ],
        "review": ["a#p0: S5 flagged"],
    }
    wb = export.build_counts_workbook(tables)
    ws = wb["Summary"]
    assert [c.value for c in ws[1]] == ["Component", "Count", "Unit"]
    assert [c.value for c in ws[2]] == ["Sprinkler head", 59, "EA"]
    assert [c.value for c in ws[3]] == ["Floor drain", 3, "EA"]
    ws2 = wb["Detail"]
    assert ws2["A1"].value == "Sheet" and ws2["B1"].value == "Cluster"
    assert ws2["B2"].value == "S0" and ws2["H2"].value == 47
    ws3 = wb["Review"]
    assert ws3["A2"].value == "a#p0: S5 flagged"
