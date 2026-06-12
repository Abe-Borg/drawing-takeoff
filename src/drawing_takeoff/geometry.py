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
from .models import GeometryPath, Network, SheetGeometry, SheetRef, TextWord

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


def extract_page_pdf(pdf_path: str, page_index: int = 0) -> bytes:
    """Extract one page as a standalone PDF (bytes) for native-PDF API input.

    A legend/lead sheet sent as a ``document`` block gives the model the page's
    real text layer alongside the rendered image — legends are mostly text, and
    reading it as text beats OCR-ing a 150-dpi raster. Extracting the single
    page keeps the request small regardless of the source document's size.
    """
    with fitz.open(pdf_path) as src:
        out = fitz.open()
        try:
            out.insert_pdf(src, from_page=page_index, to_page=page_index)
            return out.tobytes()
        finally:
            out.close()


# Fixed palette so a network keeps the same color across the image (and any caption).
_NETWORK_COLORS = [
    (0.85, 0.10, 0.10), (0.10, 0.35, 0.90), (0.10, 0.62, 0.20), (0.95, 0.55, 0.05),
    (0.60, 0.15, 0.80), (0.00, 0.65, 0.68), (0.90, 0.10, 0.55), (0.50, 0.45, 0.05),
]


def render_networks_png(
    pdf_path: str,
    page_index: int,
    networks: Sequence[Network],
    *,
    target_px: int = 2400,
) -> bytes:
    """Render the sheet with each given network drawn in its own color and labeled
    with its id — a *set-of-marks* overlay for the M7 recognition step.

    The model points by the engine's network id (drawn on the image), so its
    labels bind straight back to exact geometry. The caller passes the networks to
    show (normally the largest few); colors cycle through ``_NETWORK_COLORS``.
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        for i, nw in enumerate(networks):
            color = _NETWORK_COLORS[i % len(_NETWORK_COLORS)]
            for run in nw.runs:
                pts = [fitz.Point(*p) for p in run.polyline]
                if len(pts) >= 2:
                    page.draw_polyline(pts, color=color, width=5.0)
            x0, y0, _, _ = nw.bbox
            page.insert_text(fitz.Point(x0, max(y0 + 22, 26)), nw.id, fontsize=34, color=color)
        scale = target_px / max(page.rect.width, page.rect.height)
        return page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")


def render_network_crop_png(
    pdf_path: str,
    page_index: int,
    network: Network,
    *,
    margin_pt: float = 72.0,
    target_px: int = 1600,
) -> bytes:
    """High-DPI close-up of one network — the second-look render for M7.

    The network is highlighted (thin, so tees and fittings stay visible under
    the stroke) and the raster is clipped to its bbox plus ``margin_pt`` of
    surrounding context. Scaling the *crop* to ``target_px`` is what buys the
    resolution: a tight crop of an E-size sheet lands at many times the
    whole-sheet render's effective DPI — enough to read branch connections and
    nearby callouts that the global set-of-marks view blurs away.
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        color = _NETWORK_COLORS[0]
        for run in network.runs:
            pts = [fitz.Point(*p) for p in run.polyline]
            if len(pts) >= 2:
                page.draw_polyline(pts, color=color, width=2.0)
        x0, y0, x1, y1 = network.bbox
        clip = fitz.Rect(x0 - margin_pt, y0 - margin_pt, x1 + margin_pt, y1 + margin_pt) & page.rect
        page.insert_text(fitz.Point(clip.x0 + 6, clip.y0 + 20), network.id, fontsize=16, color=color)
        scale = target_px / max(clip.width, clip.height, 1.0)
        return page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False).tobytes("png")


def render_symbol_crop_png(
    pdf_path: str,
    page_index: int,
    bbox: tuple[float, float, float, float],
    *,
    margin_pt: float = 36.0,
    target_px: int = 512,
    highlight: bool = True,
) -> bytes:
    """High-DPI close-up of one symbol instance — the M10 exemplar crop.

    The same crop-to-target trick as :func:`render_network_crop_png`: scaling a
    tight crop of an E-size sheet buys many times the whole-sheet render's
    effective DPI, so one fixture lands at a few hundred legible pixels instead
    of a 20-pixel smudge. ``highlight`` outlines the instance bbox so the model
    knows exactly which marks ARE the symbol — everything else in the crop is
    context (the counter run vs. the wall is what disambiguates a lav from a
    urinal), not part of it.
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        x0, y0, x1, y1 = bbox
        if highlight:
            page.draw_rect(fitz.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2),
                           color=_NETWORK_COLORS[0], width=1.2)
        clip = fitz.Rect(x0 - margin_pt, y0 - margin_pt, x1 + margin_pt, y1 + margin_pt) & page.rect
        scale = target_px / max(clip.width, clip.height, 1.0)
        return page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False).tobytes("png")


def write_count_markup_pdf(
    pdf_path: str,
    page_index: int,
    clusters: Sequence,
    out_path: str,
    *,
    labels: dict | None = None,
) -> None:
    """Box every instance of every cluster (colored per cluster) and save a new
    PDF — the count takeoff's trust artifact: a double count or a missed fixture
    is visible at a glance. Each cluster is captioned at its exemplar with its
    id, count, and (when ``labels`` is given) the component name — flagged
    clusters get a trailing ``?`` so the review set stands out on paper.
    Vector marks on the original page, crisp at any zoom.
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        for i, c in enumerate(clusters):
            color = _NETWORK_COLORS[i % len(_NETWORK_COLORS)]
            for (x0, y0, x1, y1) in c.instance_bboxes:
                page.draw_rect(fitz.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2), color=color, width=1.0)
            lab = (labels or {}).get(c.id)
            caption = f"{c.id} x{c.count}"
            if lab is not None:
                caption += f" {lab.system}" + ("" if lab.trusted else " ?")
            ex0, ey0, _, _ = c.exemplar_bbox
            page.insert_text(fitz.Point(ex0, max(ey0 - 6, 10)), caption, fontsize=10, color=color)
        doc.save(out_path, garbage=3, deflate=True)


def write_marked_up_pdf(pdf_path: str, page_index: int, networks: Sequence[Network], out_path: str) -> None:
    """Draw the given networks (colored + numbered) onto the sheet and save a new
    PDF — the marked-up takeoff an estimator opens, prints, and checks the colored
    runs against. Vector marks on the original page (not a raster), so it stays
    crisp at any zoom.
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        for i, nw in enumerate(networks):
            color = _NETWORK_COLORS[i % len(_NETWORK_COLORS)]
            for run in nw.runs:
                pts = [fitz.Point(*p) for p in run.polyline]
                if len(pts) >= 2:
                    page.draw_polyline(pts, color=color, width=3.0)
            x0, y0, _, _ = nw.bbox
            page.insert_text(fitz.Point(x0, max(y0 + 18, 20)), nw.id, fontsize=22, color=color)
        doc.save(out_path, garbage=3, deflate=True)
