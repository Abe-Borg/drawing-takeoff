"""Discipline auto-detection: the keyword scorer, the pipeline resolver, and
the Claude text-classification fallback (hermetic — fake client, no network).

The contract under test: an explicit user choice wins outright; otherwise each
PDF (assumed single-discipline) is detected from its own text for free, the
LLM fallback fires only when the keywords are inconclusive AND there is enough
text to read, and nothing here can ever sink a run — the worst case is the
historical ``"construction"`` default.
"""
from __future__ import annotations

import json

import pytest

from drawing_takeoff import legend, pipeline
from drawing_takeoff.discipline import DISCIPLINES, detect_discipline
from drawing_takeoff.models import SheetGeometry, SheetRef, TextWord
from tests.fixtures.fake_anthropic import FakeClient, FakeMessage, FakeTextBlock


# ---------------------------------------------------------------------------
# detect_discipline — the deterministic keyword scorer
# ---------------------------------------------------------------------------
def test_filename_alone_detects():
    guess = detect_discipline(["", ""], filename="Weld_County_Mechanical_Permit_Set.pdf")
    assert guess is not None and guess.discipline == "Mechanical"
    assert "filename" in guess.evidence


def test_titles_and_sheet_numbers_detect_fire_protection():
    pages = [
        "FP2.20 FIRE PROTECTION ENLARGED FLOOR PLAN EAST",
        "FP2.21 FIRE PROTECTION ENLARGED FLOOR PLAN SOUTH",
    ]
    guess = detect_discipline(pages, filename="set.pdf")
    assert guess is not None and guess.discipline == "Fire Protection"
    assert "named on 2 page(s)" in guess.evidence
    assert "sheet numbers on 2 page(s)" in guess.evidence


def test_vocabulary_alone_detects():
    # No discipline named anywhere — terminology carries it.
    pages = ["SUPPLY DUCT 12x10 450 CFM", "DIFFUSER SCHEDULE AHU-1 VAV-3 DAMPER"]
    guess = detect_discipline(pages)
    assert guess is not None and guess.discipline == "Mechanical"


def test_word_boundaries_block_substring_hits():
    # CONDUCTOR/PRODUCT must not score DUCT; ELECTROMECHANICAL must not score
    # MECHANICAL. With no real evidence the detector stays inconclusive.
    assert detect_discipline(["CONDUCTOR PRODUCT ELECTROMECHANICAL AQUEDUCT"]) is None


def test_contested_evidence_is_inconclusive():
    assert detect_discipline(["MECHANICAL ROOM", "ELECTRICAL ROOM"]) is None


def test_no_text_is_inconclusive():
    assert detect_discipline([]) is None
    assert detect_discipline(["", "   "]) is None


def test_cross_reference_noise_does_not_flip_detection():
    # Mechanical set whose notes reference other trades — the per-page strong
    # evidence plus sheet numbers must keep the true discipline on top.
    pages = [
        "M1.1 MECHANICAL FLOOR PLAN SUPPLY DUCT 450 CFM",
        "M1.2 MECHANICAL ROOF PLAN RTU-1 REFER TO ELECTRICAL DRAWINGS",
        "M2.0 MECHANICAL DETAILS COORDINATE WITH PLUMBING AND ARCHITECTURAL",
    ]
    guess = detect_discipline(pages, filename="drawings.pdf")
    assert guess is not None and guess.discipline == "Mechanical"


def test_sheet_number_shapes():
    from drawing_takeoff.discipline import _sheet_number_discipline

    assert _sheet_number_discipline("M1.1") == "Mechanical"
    assert _sheet_number_discipline("m-101") == "Mechanical"
    assert _sheet_number_discipline("FP2.20") == "Fire Protection"  # not Plumbing
    assert _sheet_number_discipline("P201,") == "Plumbing"  # trailing punctuation
    assert _sheet_number_discipline("E1.1") == "Electrical"
    assert _sheet_number_discipline("12'-0") is None
    assert _sheet_number_discipline("ROOM") is None
    assert _sheet_number_discipline("M") is None


# ---------------------------------------------------------------------------
# legend.classify_discipline — the Claude text fallback
# ---------------------------------------------------------------------------
def _classify_client(reply: dict):
    return FakeClient(lambda kw: FakeMessage(content=[FakeTextBlock(text=json.dumps(reply))]))


def test_classify_discipline_returns_enum_choice():
    client = _classify_client(
        {"discipline": "Plumbing", "confidence": "high", "reasoning": "P-sheet titles"}
    )
    got = legend.classify_discipline(
        ["sanitary riser diagram"], filename="set.pdf", client=client
    )
    assert got == ("Plumbing", "P-sheet titles")
    # The request is schema-enforced and carries the filename + page text.
    kw = client.messages.calls[0]
    schema = kw["output_config"]["format"]["schema"]
    assert set(schema["properties"]["discipline"]["enum"]) == {*DISCIPLINES, "unknown"}
    text = " ".join(b["text"] for b in kw["messages"][0]["content"] if b.get("type") == "text")
    assert "set.pdf" in text and "sanitary riser diagram" in text


def test_classify_discipline_unknown_or_garbage_is_none():
    unknown = _classify_client({"discipline": "unknown", "confidence": "low", "reasoning": ""})
    assert legend.classify_discipline(["x"], client=unknown) is None
    garbage = FakeClient(lambda kw: FakeMessage(content=[FakeTextBlock(text="not json")]))
    assert legend.classify_discipline(["x"], client=garbage) is None


# ---------------------------------------------------------------------------
# pipeline._resolve_disciplines — override > scorer > Claude > default
# ---------------------------------------------------------------------------
def _sheet(source: str, text: str, page: int = 0) -> SheetGeometry:
    words = [TextWord(t, (0.0, 0.0, 1.0, 1.0)) for t in text.split()]
    return SheetGeometry(
        ref=SheetRef(source, page), page_width_pt=612, page_height_pt=792,
        paths=[], words=words,
    )


def _boom_client():
    def responder(kw):
        raise AssertionError("the client must not be called")

    return FakeClient(responder)


def test_resolver_explicit_override_wins():
    sheets = [_sheet("m.pdf", "MECHANICAL FLOOR PLAN M1.1")]
    out = pipeline._resolve_disciplines(
        sheets, requested="Electrical", client=_boom_client(), log=lambda m: None
    )
    assert out == {"m.pdf": "Electrical"}


def test_resolver_detects_per_pdf_without_a_client_call():
    logs: list[str] = []
    sheets = [
        _sheet("m.pdf", "M1.1 MECHANICAL FLOOR PLAN SUPPLY DUCT CFM"),
        _sheet("m.pdf", "M1.2 MECHANICAL ROOF PLAN RTU-1", page=1),
        _sheet("fp.pdf", "FP2.20 FIRE PROTECTION ENLARGED FLOOR PLAN"),
    ]
    out = pipeline._resolve_disciplines(
        sheets, requested=None, client=_boom_client(), log=logs.append
    )
    assert out == {"m.pdf": "Mechanical", "fp.pdf": "Fire Protection"}
    assert any("m.pdf" in m and "Mechanical" in m for m in logs)


def test_resolver_falls_back_to_claude_when_keywords_inconclusive():
    # Plenty of text, zero discipline markers -> the classification call fires.
    neutral = "general notes apply to all sheets verify dimensions in field " * 8
    sheets = [_sheet("x.pdf", neutral)]
    client = _classify_client(
        {"discipline": "Structural", "confidence": "medium", "reasoning": "framing terms"}
    )
    logs: list[str] = []
    out = pipeline._resolve_disciplines(sheets, requested=None, client=client, log=logs.append)
    assert out == {"x.pdf": "Structural"}
    assert len(client.messages.calls) == 1
    assert any("classified by Claude" in m and "Structural" in m for m in logs)


def test_resolver_skips_claude_on_scanned_like_sheets():
    # Almost no text (a scanned/raster set) -> nothing for the LLM to read
    # either; no call is burned and the neutral default applies.
    logs: list[str] = []
    out = pipeline._resolve_disciplines(
        [_sheet("scan.pdf", "tiny note")], requested=None, client=_boom_client(), log=logs.append
    )
    assert out == {"scan.pdf": "construction"}
    assert any("could not auto-detect" in m for m in logs)


def test_resolver_survives_classification_failure():
    neutral = "general notes apply to all sheets verify dimensions in field " * 8

    def responder(kw):
        raise RuntimeError("api down")

    out = pipeline._resolve_disciplines(
        [_sheet("x.pdf", neutral)], requested=None, client=FakeClient(responder),
        log=lambda m: None,
    )
    assert out == {"x.pdf": "construction"}


# ---------------------------------------------------------------------------
# End to end: the detected discipline anchors the labeling prompt
# ---------------------------------------------------------------------------
def test_extract_takeoff_threads_detected_discipline_into_labeling(tmp_path):
    fitz = pytest.importorskip("fitz")

    p = tmp_path / "sheet.pdf"
    doc = fitz.open()
    page = doc.new_page(width=216, height=144)
    shape = page.new_shape()
    shape.draw_line(fitz.Point(20, 50), fitz.Point(110, 50))
    shape.finish(color=(0, 0, 0), width=1.3)
    shape.commit()
    page.insert_text(fitz.Point(20, 20), '1/8" = 1\'-0"', fontsize=8)
    page.insert_text(fitz.Point(20, 35), "M1.1 MECHANICAL FLOOR PLAN DUCT CFM", fontsize=8)
    doc.save(p)
    doc.close()

    contents = []

    def responder(kw):
        contents.append(kw["messages"][0]["content"])
        return FakeMessage(content=[FakeTextBlock(text=json.dumps({"labels": [
            {"style_id": "s0", "system": "Supply duct", "measurable": True,
             "confidence": "high", "ambiguous": False, "reasoning": ""}]}))])

    result = pipeline.extract_takeoff([str(p)], client=FakeClient(responder))
    assert not result.errors
    assert len(contents) == 1  # detection was free — only the labeling call hit the client
    summary = contents[0][0]["text"]
    assert "Sheet discipline: Mechanical." in summary
