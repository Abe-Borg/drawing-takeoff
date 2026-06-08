"""Scale handling: label -> points_per_foot, plus a known-dimension verifier.

The user constraint that makes this arithmetic instead of calibration: drawings
are true-to-scale and every sheet is labeled with its scale. So::

    points_per_foot = 72 * (paper_inches_per_foot)

e.g. 1/4"=1'-0"  ->  72 * 0.25  = 18 pt/ft
     1/8"=1'-0"  ->  72 * 0.125 =  9 pt/ft
     1"=20'-0"   ->  72 * (1/20) = 3.6 pt/ft   (engineering scale)

and ``feet = measured_points / points_per_foot``. Nothing here imports PyMuPDF;
it operates on the scale-label string the geometry layer reads off the sheet.
"""
from __future__ import annotations

import re

_PT_PER_INCH = 72.0

# Unicode prime / double-prime / curly quotes -> ASCII ' and ", and the common
# vulgar fractions, so a label lifted from a PDF normalizes before parsing.
_PRIME = {"′": "'", "’": "'", "´": "'", "`": "'"}
_DPRIME = {"″": '"', "”": '"', "“": '"'}
_VULGAR = {
    "¼": "1/4", "½": "1/2", "¾": "3/4",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
    "⅓": "1/3", "⅔": "2/3",
}


def _normalize(text: str) -> str:
    for src, dst in {**_PRIME, **_DPRIME, **_VULGAR}.items():
        text = text.replace(src, dst)
    return text


def parse_feet_inches(token: str, *, allow_bare_hyphen: bool = False) -> float | None:
    """Parse a feet-inches token to a float number of feet, or ``None``.

    Handles ``12'-0"``, ``12'``, ``10'-10"`` and unicode prime/quote variants.
    With ``allow_bare_hyphen`` it also accepts the fire-sprinkler shorthand
    ``12-0`` / ``10-10`` (feet-hyphen-inches with no marks) used to tag pipe
    runs on this sheet — kept opt-in because a bare ``N-N`` is ambiguous with
    node ids elsewhere on the drawing.
    """
    s = _normalize(token).strip()
    # feet with an explicit foot mark, optional inches
    m = re.fullmatch(r"(\d+)\s*'\s*(?:-\s*(\d+(?:\.\d+)?)\s*\"?)?", s)
    if m:
        ft = float(m.group(1))
        inch = float(m.group(2)) if m.group(2) else 0.0
        return ft + inch / 12.0 if inch < 12 else None
    if allow_bare_hyphen:
        m = re.fullmatch(r"(\d+)-(\d+)", s)
        if m:
            inch = int(m.group(2))
            return int(m.group(1)) + inch / 12.0 if inch < 12 else None
    return None


def _parse_paper_inches(token: str) -> float | None:
    """Parse the left side of a scale label (paper measure) to inches.

    Accepts a whole number (``3``), a simple fraction (``1/8``, ``3/32``), a
    mixed number (``1-1/2``), or a decimal — with an optional trailing ``"``.
    """
    s = _normalize(token).strip().rstrip('"').strip()
    m = re.fullmatch(r"(\d+)-(\d+)/(\d+)", s)        # mixed: 1-1/2
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.fullmatch(r"(\d+)/(\d+)", s)              # fraction: 1/8
    if m:
        return int(m.group(1)) / int(m.group(2))
    m = re.fullmatch(r"\d+(?:\.\d+)?", s)            # whole / decimal
    if m:
        return float(s)
    return None


# A scale label is "<paper>" = "<ground feet>" — e.g. 1/8" = 1'-0", 1" = 20'-0".
# The ground side keeps its closing inch mark so the echoed label reads whole.
_SCALE_RE = re.compile(
    r'(?P<paper>\d[\d\s./-]*?)\s*"\s*=\s*(?P<ground>\d+\s*\'(?:\s*-\s*\d+\s*"?)?)'
)


def points_per_foot_from_label(scale_label: str) -> float:
    """Convert a scale label to PDF ``points_per_foot``.

    Raises :class:`ValueError` if the label cannot be parsed — the caller should
    surface that (or fall back to a GUI scale-confirm field) rather than measure
    against a guessed scale.
    """
    norm = _normalize(scale_label)
    m = _SCALE_RE.search(norm)
    if not m:
        raise ValueError(f"Unrecognized scale label: {scale_label!r}")
    paper_in = _parse_paper_inches(m.group("paper"))
    ground_ft = parse_feet_inches(m.group("ground"))
    if paper_in is None or not ground_ft:
        raise ValueError(f"Unrecognized scale label: {scale_label!r}")
    return _PT_PER_INCH * paper_in / ground_ft


def detect_scale_label(text: str) -> str | None:
    """Find a scale label in free page text, or ``None``.

    Returns the matched core (e.g. ``'1/8" = 1\\'-0"'``) only if it actually
    parses, so a stray ``= 1'`` elsewhere on the sheet is not mistaken for one.
    """
    norm = _normalize(text)
    for m in _SCALE_RE.finditer(norm):
        core = m.group(0).strip()
        try:
            points_per_foot_from_label(core)
        except ValueError:
            continue
        return core
    return None


def verify_against_dimension(measured_pt: float, stated_ft: float, ppf: float) -> float:
    """Return the signed percent error of a measured length vs. a known value.

    ``measured_pt / ppf`` is the measured length in feet; the result is
    ``100 * (measured_ft - stated_ft) / stated_ft``. The M1 gate wants this
    under ~1% on a known dimension before any total is trusted. Positive = the
    geometry measured longer than the stated dimension.
    """
    if stated_ft == 0 or ppf == 0:
        raise ValueError("stated_ft and ppf must be non-zero")
    measured_ft = measured_pt / ppf
    return 100.0 * (measured_ft - stated_ft) / stated_ft
