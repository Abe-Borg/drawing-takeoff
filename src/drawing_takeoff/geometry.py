"""The ONLY PyMuPDF module — the AGPL boundary lives here.

Extracts vector paths (``page.get_drawings()``) and text-with-coords
(``page.get_text("words")``) off a page and hands back pure
:mod:`drawing_takeoff.models` shapes that carry no ``fitz`` types. Every other
module operates on those shapes, so the backend stays swappable and the license
boundary stays clean. Importing ``fitz`` anywhere outside this module is a bug.

The later legend/recognition crops (M3) will add a small rasterizer here; for
now this is the geometry + text reader the M1 scale proof needs.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import fitz  # PyMuPDF (AGPL-3.0) — confined to this module by design.

from . import scale as _scale
from .models import GeometryPath, SheetGeometry, SheetRef, TextWord

# PyMuPDF path-type code -> our ``kind``.
_KIND = {"s": "stroke", "f": "fill", "fs": "both"}


def _pt(p: "fitz.Point") -> tuple[float, float]:
    return (float(p.x), float(p.y))


def _normalize_item(item: tuple) -> tuple:
    """Convert one ``get_drawings`` item to a backend-free segment tuple.

    Shapes (PyMuPDF): ``("l", p0, p1)``, ``("c", p0, p1, p2, p3)``,
    ``("re", rect, orientation?)``, ``("qu", quad)``. fitz points/rects/quads
    become plain float tuples; a trailing orientation int (newer ``re``) is
    dropped — length/area math never uses it.
    """
    op = item[0]
    if op == "l":
        return ("l", _pt(item[1]), _pt(item[2]))
    if op == "c":
        return ("c", _pt(item[1]), _pt(item[2]), _pt(item[3]), _pt(item[4]))
    if op == "re":
        r = item[1]
        return ("re", (float(r.x0), float(r.y0), float(r.x1), float(r.y1)))
    if op == "qu":
        q = item[1]
        return ("qu", _pt(q.ul), _pt(q.ur), _pt(q.ll), _pt(q.lr))
    # Unknown op: keep the code, stringify the rest defensively.
    return (op, *item[1:])  # pragma: no cover - defensive


def _color(value) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(c) for c in value)


def _path_from_drawing(d: dict) -> GeometryPath:
    r = d["rect"]
    return GeometryPath(
        items=tuple(_normalize_item(it) for it in d["items"]),
        stroke_color=_color(d.get("color")),
        fill_color=_color(d.get("fill")),
        width=float(d["width"]) if d.get("width") is not None else None,
        dashes=d.get("dashes") or "[] 0",
        closed=bool(d.get("closePath", False)),
        bbox=(float(r.x0), float(r.y0), float(r.x1), float(r.y1)),
        kind=_KIND.get(d.get("type"), d.get("type") or "stroke"),
    )


def _words(page: "fitz.Page") -> list[TextWord]:
    out: list[TextWord] = []
    for (x0, y0, x1, y1, text, *_rest) in page.get_text("words"):
        out.append(TextWord(text=text, bbox=(float(x0), float(y0), float(x1), float(y1))))
    return out


def extract_sheet_geometry(page: "fitz.Page", *, ref: SheetRef | None = None) -> SheetGeometry:
    """Extract one page's vector paths + words into a :class:`SheetGeometry`.

    The scale label is auto-detected from the page text and resolved to
    ``points_per_foot`` when parseable; callers (GUI scale-confirm field,
    pipeline) may override either afterward.
    """
    if ref is None:
        ref = SheetRef(source=getattr(page.parent, "name", "<page>") or "<page>",
                       page_index=page.number)
    rect = page.rect
    paths = [_path_from_drawing(d) for d in page.get_drawings()]
    words = _words(page)

    geom = SheetGeometry(
        ref=ref,
        page_width_pt=float(rect.width),
        page_height_pt=float(rect.height),
        paths=paths,
        words=words,
    )
    label = _scale.detect_scale_label(page.get_text("text"))
    if label:
        geom.scale_label = label
        try:
            geom.points_per_foot = _scale.points_per_foot_from_label(label)
        except ValueError:  # pragma: no cover - detect_scale_label pre-validates
            geom.points_per_foot = None
    return geom


def extract_pdf_geometry(
    pdf_path: str, *, pages: Sequence[int] | None = None
) -> list[SheetGeometry]:
    """Open a PDF and extract geometry for ``pages`` (default: all).

    The headless entry point the diagnostics script and (M4) pipeline use, so
    no caller outside this module ever opens a ``fitz`` document.
    """
    out: list[SheetGeometry] = []
    with fitz.open(pdf_path) as doc:
        indices: Iterable[int] = pages if pages is not None else range(doc.page_count)
        for i in indices:
            page = doc[i]
            ref = SheetRef(source=pdf_path, page_index=i)
            out.append(extract_sheet_geometry(page, ref=ref))
    return out


# ---------------------------------------------------------------------------
# Rasterizers for the M3 legend/recognition step (PNG bytes -> legend.py)
# ---------------------------------------------------------------------------
def render_style_swatch(style_key, *, width_px: int = 240, height_px: int = 48, dpi: float = 2.0) -> bytes:
    """Render a sample line drawn in ``style_key``'s pen as PNG bytes.

    Lets the recognition model *see* what a style looks like (heavy black solid
    vs. thin gray dashed) alongside its stats. Uses a synthetic one-line page so
    no source PDF is needed.
    """
    doc = fitz.open()
    w_pt, h_pt = width_px / dpi, height_px / dpi
    page = doc.new_page(width=w_pt, height=h_pt)
    shape = page.new_shape()
    y = h_pt / 2.0
    shape.draw_line(fitz.Point(w_pt * 0.08, y), fitz.Point(w_pt * 0.92, y))
    color = style_key.stroke_color or (0.0, 0.0, 0.0)
    shape.finish(color=tuple(color), width=max(style_key.width or 0.3, 0.3), dashes=style_key.dashes)
    shape.commit()
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi, dpi), alpha=False)
    data = pix.tobytes("png")
    doc.close()
    return data


def image_file_to_png(path: str) -> bytes:
    """Load a raster image file and re-encode it as PNG bytes.

    Normalizes ``.jpg`` / ``.webp`` / ``.gif`` / ... so a legend image passed to
    the recognition step is honestly labeled ``image/png`` (the request always
    advertises PNG). PyMuPDF's PNG encoder only handles gray/RGB, so a CMYK scan
    (common for legend sheets) is converted to RGB first — otherwise
    ``tobytes("png")`` raises.
    """
    pix = fitz.Pixmap(path)
    if pix.colorspace is not None and pix.colorspace.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix.tobytes("png")


def render_page_png(
    pdf_path: str,
    page_index: int = 0,
    *,
    dpi: int = 150,
    clip: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Rasterize a page (or a ``clip`` region of it) to PNG bytes.

    Used to hand the recognition model a legend/symbols image off the lead
    sheet, or a crop of the drawing for context.
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        clip_rect = fitz.Rect(*clip) if clip is not None else None
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        return pix.tobytes("png")
