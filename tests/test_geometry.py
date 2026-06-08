"""PyMuPDF-gated test: extract geometry from a tiny synthetic vector PDF.

Mirrors the sibling project's render-test gating — skipped wholesale when
PyMuPDF is unavailable. Builds a one-line sheet of known length (90 pt = 10 ft
at 1/8"=1'-0") so the extraction can be checked against exact numbers without
committing a binary fixture.
"""
from __future__ import annotations

import math

import pytest

fitz = pytest.importorskip("fitz")  # skip the whole module if PyMuPDF is absent

from drawing_takeoff import geometry
from drawing_takeoff.models import SheetGeometry


@pytest.fixture
def synthetic_pdf(tmp_path):
    """A 3x2in page: one 90pt black line, a scale label, and a pipe tag."""
    path = tmp_path / "synthetic.pdf"
    doc = fitz.open()
    page = doc.new_page(width=216, height=144)  # 3 x 2 inches
    shape = page.new_shape()
    shape.draw_line(fitz.Point(20, 50), fitz.Point(110, 50))  # 90 pt long
    shape.finish(color=(0, 0, 0), width=1.3)
    shape.commit()
    page.insert_text(fitz.Point(20, 20), '1/8" = 1\'-0"', fontsize=8)
    page.insert_text(fitz.Point(55, 45), "10-0", fontsize=6)
    doc.save(path)
    doc.close()
    return str(path)


def test_extract_pdf_geometry_returns_pure_models(synthetic_pdf):
    geoms = geometry.extract_pdf_geometry(synthetic_pdf)
    assert len(geoms) == 1
    geom = geoms[0]
    assert isinstance(geom, SheetGeometry)
    assert geom.page_width_pt == pytest.approx(216)
    assert geom.page_height_pt == pytest.approx(144)
    assert geom.ref.page_index == 0
    # No fitz types leak into the models.
    for p in geom.paths:
        for it in p.items:
            for el in it[1:]:
                assert isinstance(el, tuple)


def test_extracts_the_known_line(synthetic_pdf):
    geom = geometry.extract_pdf_geometry(synthetic_pdf)[0]
    line = None
    for p in geom.paths:
        for it in p.items:
            if it[0] == "l":
                length = math.dist(it[1], it[2])
                if length > 50:  # the 90pt line, not glyph strokes
                    line = (it, p, length)
    assert line is not None, "the drawn 90pt line was not extracted"
    it, path, length = line
    assert length == pytest.approx(90.0, abs=0.5)
    assert path.width == pytest.approx(1.3, abs=0.05)
    assert path.stroke_color == pytest.approx((0.0, 0.0, 0.0), abs=0.01)


def test_scale_label_auto_detected(synthetic_pdf):
    geom = geometry.extract_pdf_geometry(synthetic_pdf)[0]
    assert geom.scale_label is not None
    assert geom.points_per_foot == pytest.approx(9.0)


def test_words_carry_text_and_coords(synthetic_pdf):
    geom = geometry.extract_pdf_geometry(synthetic_pdf)[0]
    texts = {w.text for w in geom.words}
    assert "10-0" in texts
    # the line of geometry is not text; words are only the inserted strings
    assert any("1/8" in t for t in texts)
