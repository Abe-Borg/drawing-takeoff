"""Dependency-free data models for the takeoff engine.

These shapes carry geometry and text out of :mod:`drawing_takeoff.geometry`
(the only PyMuPDF module) into the pure-Python measurement / scale / export
layers. Nothing here imports PyMuPDF: coordinates are plain ``(x, y)`` float
tuples, not ``fitz.Point``, so ``measure``/``scale``/``pipeline``/``export`` and
the tests can manipulate geometry without the AGPL backend on the path.

Only the shapes the current milestones need are defined. ``Run`` (M2),
``TakeoffItem`` and ``TakeoffResult`` (M4) land with their milestones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

# Rounding precision for the style-grouping key. Construction linework for one
# system is drawn with a single pen, so colors/widths repeat to many decimals;
# rounding collapses float jitter without merging genuinely distinct pens.
_COLOR_NDIGITS = 3
_WIDTH_NDIGITS = 2

# A normalized path segment: a tuple whose first element is the op code
# ("l", "c", "re", "qu") followed by plain coordinate tuples (never fitz types).
Point = tuple[float, float]
Segment = tuple  # ("l", p0, p1) | ("c", p0, p1, p2, p3) | ("re", bbox) | ("qu", ...)
BBox = tuple[float, float, float, float]


class SheetRef(NamedTuple):
    """Identifies a source page (mirrors the sibling project's ``SheetRef``)."""

    source: str        # PDF path or filename
    page_index: int    # 0-based page index

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.source}#p{self.page_index}"


class StyleKey(NamedTuple):
    """Hashable exact-style grouping key: ``(stroke_color, width, dashes)``.

    The propagation mechanism for labeling: every path drawn with one pen
    shares a key, so classifying one style labels all its paths. Build via
    :meth:`from_path_attrs` so rounding is applied consistently.
    """

    stroke_color: tuple[float, ...] | None
    width: float
    dashes: str

    @classmethod
    def from_path_attrs(
        cls,
        stroke_color: tuple[float, ...] | None,
        width: float | None,
        dashes: str | None,
    ) -> "StyleKey":
        color = (
            tuple(round(c, _COLOR_NDIGITS) for c in stroke_color)
            if stroke_color is not None
            else None
        )
        return cls(color, round(width or 0.0, _WIDTH_NDIGITS), dashes or "[] 0")


@dataclass(frozen=True)
class TextWord:
    """One word from ``page.get_text("words")`` with its bbox."""

    text: str
    bbox: BBox

    @property
    def centroid(self) -> Point:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass(frozen=True)
class GeometryPath:
    """One vector path from ``page.get_drawings()``, backend-free.

    ``items`` are normalized segment tuples (see :data:`Segment`). ``kind`` is
    ``"stroke"`` / ``"fill"`` / ``"both"`` (PyMuPDF ``s`` / ``f`` / ``fs``).
    """

    items: tuple[Segment, ...]
    stroke_color: tuple[float, ...] | None
    fill_color: tuple[float, ...] | None
    width: float | None
    dashes: str
    closed: bool
    bbox: BBox
    kind: str

    @property
    def style_key(self) -> StyleKey:
        return StyleKey.from_path_attrs(self.stroke_color, self.width, self.dashes)

    @property
    def bbox_width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def is_degenerate(self) -> bool:
        """A zero-extent path (the sheet's ~84k stipple artifacts) — drop it."""
        return self.bbox_width <= 0.0 and self.bbox_height <= 0.0


@dataclass
class SheetGeometry:
    """Everything measurement needs from one page, with no PyMuPDF on the path."""

    ref: SheetRef
    page_width_pt: float
    page_height_pt: float
    paths: list[GeometryPath] = field(default_factory=list)
    words: list[TextWord] = field(default_factory=list)
    scale_label: str | None = None
    points_per_foot: float | None = None

    def non_degenerate_paths(self) -> list[GeometryPath]:
        """Paths with real extent (drops the zero-length stipple artifacts)."""
        return [p for p in self.paths if not p.is_degenerate]


@dataclass(frozen=True)
class Run:
    """A stitched, maximally-straight run of one style — the unit of measurement.

    Fragments the CAD export split along one straight line are stitched back
    into a single ``Run`` (see :func:`drawing_takeoff.measure.stitch_runs`), so
    ``length_pt`` is the honest drawn length and the per-style total is a sum of
    runs rather than fragments. ``length_ft`` is filled when a ``points_per_foot``
    is known. A direction change (elbow) or junction (tee) starts a new run.
    """

    style_key: StyleKey
    polyline: tuple[Point, ...]
    length_pt: float
    bbox: BBox
    length_ft: float | None = None
    segment_count: int = 1

    @property
    def centroid(self) -> Point:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass(frozen=True)
class Network:
    """A connected component of runs — the M5 unit that maps to one physical
    system. Built by :func:`drawing_takeoff.measure.networks` via tee-aware
    (endpoint-to-segment) connectivity, so a branch that tees into the *middle*
    of a main joins the same network — the way an estimator "follows the line".
    A network may span lineweights (a main and its branches can differ), so it is
    deliberately *not* keyed on style. Derived quantities are properties, so a
    network stays defined by its runs alone.
    """

    id: str
    runs: tuple[Run, ...]
    ppf: float | None = None

    @property
    def run_count(self) -> int:
        return len(self.runs)

    @property
    def length_pt(self) -> float:
        return sum(r.length_pt for r in self.runs)

    @property
    def length_ft(self) -> float | None:
        return (self.length_pt / self.ppf) if self.ppf else None

    @property
    def bbox(self) -> BBox:
        return (
            min(r.bbox[0] for r in self.runs), min(r.bbox[1] for r in self.runs),
            max(r.bbox[2] for r in self.runs), max(r.bbox[3] for r in self.runs),
        )

    @property
    def style_keys(self) -> tuple[StyleKey, ...]:
        """Distinct styles present — a network can cross lineweights."""
        return tuple(dict.fromkeys(r.style_key for r in self.runs))


@dataclass(frozen=True)
class SystemLabel:
    """What a line style *means* — the output of the M3 legend/recognition step.

    Construction legends are unreliable (incomplete, on a different sheet, or
    contradicted by the drawing), so a label carries its own ``confidence`` and
    an explicit ``ambiguous`` flag: the engine totals only styles it can stand
    behind and surfaces the rest for human review rather than guessing.
    ``measurable`` separates a linear run (pipe/duct/conduit/wall) from
    background/text/symbol linework that must never enter a length total.
    """

    system: str
    measurable: bool
    confidence: str = "low"          # "high" | "medium" | "low"
    ambiguous: bool = False
    size: str | None = None
    reasoning: str | None = None

    @property
    def trusted(self) -> bool:
        """Safe to auto-total: a measurable run, not ambiguous, decent confidence."""
        return self.measurable and not self.ambiguous and self.confidence in ("high", "medium")


@dataclass(frozen=True)
class TakeoffItem:
    """One measured, named quantity for the takeoff — a style's footage on a sheet."""

    system: str
    quantity: float
    unit: str                  # "LF" for linear length
    sheet: str                 # "<source>#p<page>"
    style_key: StyleKey
    scale_used: float          # points_per_foot the measurement used
    confidence: str = "low"    # carried from the legend label
    ambiguous: bool = False
    run_count: int = 0         # provenance: how many stitched runs
    reasoning: str | None = None

    @property
    def trusted(self) -> bool:
        """Counts toward an automatic total (not ambiguous, decent confidence)."""
        return not self.ambiguous and self.confidence in ("high", "medium")


@dataclass
class TakeoffResult:
    """The end-to-end takeoff: per-sheet items, cross-sheet totals, and notes.

    ``per_system_totals`` sums only :attr:`TakeoffItem.trusted` items across all
    sheets; ambiguous styles surface via :attr:`flagged` for human review rather
    than being silently counted. ``errors`` captures per-sheet failures so one
    bad sheet never sinks the run; ``diagnostics`` is a human-readable trail.
    """

    items: list[TakeoffItem] = field(default_factory=list)
    per_system_totals: dict[str, float] = field(default_factory=dict)
    sheet_count: int = 0
    errors: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def trusted_items(self) -> list[TakeoffItem]:
        return [i for i in self.items if i.trusted]

    @property
    def flagged(self) -> list[TakeoffItem]:
        """Measurable but not trusted — surfaced for confirmation, not counted."""
        return [i for i in self.items if not i.trusted]
