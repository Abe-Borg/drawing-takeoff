"""Discipline auto-detection: which trade a drawing set belongs to.

The labeling prompts anchor on a discipline string ("Sheet discipline:
Mechanical") — the prior that tells the model whose takeoff this is, hence
which line styles are scope to total and which are background (walls are the
deliverable on an architectural set and background on a mechanical one).
Asking the user for it is friction the sheets can usually absorb:
construction sets announce their trade in the filename
("..._Mechanical_Permit_Set.pdf"), the sheet numbers (M1.1 / P-201 / FP2.20),
the title block ("MECHANICAL FLOOR PLAN"), and the terminology (CFM/AHU vs.
NFPA/standpipe).

This module is the deterministic reader of that evidence — pure text scoring
over the already-extracted page words, no PyMuPDF, no network — so detection
costs nothing and is unit-testable. When it is inconclusive the pipeline falls
back to a small text-only Claude classification
(:func:`drawing_takeoff.legend.classify_discipline`) and finally to the
generic ``"construction"`` default; an explicit user choice beats everything.
Each PDF is assumed to carry a single discipline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

__all__ = ["DISCIPLINES", "DisciplineGuess", "detect_discipline"]

# Canonical discipline names — the GUI dropdown options and the values the
# detectors return. Order is the dropdown's display order.
DISCIPLINES: tuple[str, ...] = (
    "Mechanical",
    "Plumbing",
    "Structural",
    "Architectural",
    "Electrical",
    "Fire Protection",
)

# The discipline's own name (and near-synonyms) — the strongest text evidence.
# Matched word-bounded so ELECTROMECHANICAL never reads as MECHANICAL.
_STRONG: dict[str, tuple[str, ...]] = {
    "Mechanical": ("MECHANICAL", "HVAC"),
    "Plumbing": ("PLUMBING",),
    "Structural": ("STRUCTURAL",),
    "Architectural": ("ARCHITECTURAL", "ARCHITECTURE"),
    "Electrical": ("ELECTRICAL",),
    "Fire Protection": ("FIRE PROTECTION", "FIRE SPRINKLER", "SPRINKLER"),
}

# Discipline-specific terminology — weak, tie-breaking evidence (one point per
# distinct term seen anywhere in the set, capped). Terms are chosen to be rare
# outside their own trade's sheets; generic words (PIPE, PLAN, SCHEDULE) stay out.
_VOCAB: dict[str, tuple[str, ...]] = {
    "Mechanical": (
        "DUCT", "DUCTWORK", "CFM", "AHU", "RTU", "VAV", "DIFFUSER", "GRILLE",
        "LOUVER", "DAMPER", "CONDENSATE", "REFRIGERANT", "FURNACE", "THERMOSTAT",
        "SUPPLY AIR", "RETURN AIR", "EXHAUST FAN",
    ),
    "Plumbing": (
        "SANITARY", "CLEANOUT", "VTR", "WATER HEATER", "HOSE BIBB", "BACKFLOW",
        "DWV", "LAVATORY", "URINAL", "WATER CLOSET", "FLOOR DRAIN", "FIXTURE UNITS",
    ),
    "Structural": (
        "FOOTING", "REBAR", "REINFORCING", "ANCHOR BOLT", "SHEAR WALL", "JOIST",
        "GIRDER", "TRUSS", "FOUNDATION", "FRAMING", "CMU", "GRADE BEAM",
    ),
    "Architectural": (
        "REFLECTED CEILING", "ROOM FINISH", "DOOR SCHEDULE", "WINDOW SCHEDULE",
        "WALL TYPE", "PARTITION", "CASEWORK", "MILLWORK", "FINISH SCHEDULE",
        "GYPSUM", "STOREFRONT",
    ),
    "Electrical": (
        "PANELBOARD", "RECEPTACLE", "LUMINAIRE", "CIRCUIT BREAKER", "SWITCHBOARD",
        "SWITCHGEAR", "TRANSFORMER", "CONDUIT", "VOLTAGE", "AMPACITY", "GROUNDING",
        "LIGHTING FIXTURE",
    ),
    "Fire Protection": (
        "NFPA", "STANDPIPE", "FDC", "FIRE RISER", "PENDENT", "ESCUTCHEON",
        "WET PIPE", "DRY PIPE", "HAZARD CLASSIFICATION", "K-FACTOR",
    ),
}

# Sheet-number tokens: M1.1 / M-101 / P201 / FP2.20 / S2.0 / A101 / E1.1.
# FP must precede the single letters so FP2.20 never reads as a P sheet.
_SHEET_NO_RE = re.compile(r"^(FP|[MPSAE])[-.]?\d{1,3}(?:\.\d{1,2})?$", re.IGNORECASE)
_PREFIX_TO_DISCIPLINE: dict[str, str] = {
    "FP": "Fire Protection",
    "M": "Mechanical",
    "P": "Plumbing",
    "S": "Structural",
    "A": "Architectural",
    "E": "Electrical",
}
_TOKEN_TRIM = ",.;:()[]{}'\""

# Evidence weights. Strong/prefix evidence counts *pages* carrying it, not raw
# occurrences — a mechanical set names its trade in every title block, while a
# stray "REFER TO ELECTRICAL" note stays one page. Vocabulary counts distinct
# terms across the whole set, capped, so a 500-row CFM schedule cannot swamp.
_W_FILENAME = 12
_W_STRONG_PAGE = 4
_W_PREFIX_PAGE = 2
_VOCAB_CAP = 6

# Decision rule: the winner needs real evidence (one strong page-hit, or a
# filename hit, or several weak signals) AND a clear margin over the runner-up;
# anything murkier returns None so the caller can escalate (LLM, then default).
_MIN_SCORE = 4
_MARGIN = 1.5


def _term_pattern(term: str) -> re.Pattern[str]:
    """Word-bounded pattern for an (already uppercase) term; phrases tolerate
    any whitespace run between words."""
    return re.compile(r"\b" + r"\s+".join(re.escape(w) for w in term.split()) + r"\b")


_STRONG_RES = {d: tuple(_term_pattern(t) for t in ts) for d, ts in _STRONG.items()}
_VOCAB_RES = {d: tuple((t, _term_pattern(t)) for t in ts) for d, ts in _VOCAB.items()}


@dataclass(frozen=True)
class DisciplineGuess:
    """A confident detection: the canonical name, its score, and the evidence
    summary (human-readable, for the run log)."""

    discipline: str
    score: float
    evidence: str


def _sheet_number_discipline(token: str) -> str | None:
    """The discipline a sheet-number-shaped token points at, or ``None``."""
    m = _SHEET_NO_RE.fullmatch(token.strip(_TOKEN_TRIM))
    return _PREFIX_TO_DISCIPLINE[m.group(1).upper()] if m else None


def detect_discipline(
    page_texts: Sequence[str], *, filename: str = ""
) -> DisciplineGuess | None:
    """Detect the set's trade from its page text + filename, or ``None``.

    ``page_texts`` is one string per page (e.g. the page's words joined with
    spaces); ``filename`` is the PDF's name. Returns ``None`` when the
    evidence is absent or contested — deliberately conservative, because the
    caller has fallbacks and a wrong-but-confident discipline silently skews
    every label on the sheet.
    """
    pages = [t.upper() for t in page_texts if t and t.strip()]
    # Filenames separate words with underscores/hyphens/dots; normalize so
    # "Fire_Protection" and "Fire-Protection" match the phrase pattern.
    fname = re.sub(r"[_\-.]+", " ", filename).upper()

    prefix_pages: dict[str, set[int]] = {d: set() for d in DISCIPLINES}
    for i, text in enumerate(pages):
        for token in text.split():
            d = _sheet_number_discipline(token)
            if d:
                prefix_pages[d].add(i)

    best: list[tuple[float, str, str]] = []
    for d in DISCIPLINES:
        fn_hit = bool(fname) and any(p.search(fname) for p in _STRONG_RES[d])
        strong = sum(1 for t in pages if any(p.search(t) for p in _STRONG_RES[d]))
        prefixes = len(prefix_pages[d])
        vocab = sorted(term for term, p in _VOCAB_RES[d] if any(p.search(t) for t in pages))
        score = (
            _W_FILENAME * fn_hit
            + _W_STRONG_PAGE * strong
            + _W_PREFIX_PAGE * prefixes
            + min(len(vocab), _VOCAB_CAP)
        )
        parts = []
        if fn_hit:
            parts.append("filename")
        if strong:
            parts.append(f"named on {strong} page(s)")
        if prefixes:
            parts.append(f"sheet numbers on {prefixes} page(s)")
        if vocab:
            parts.append("terms: " + ", ".join(t.title() for t in vocab[:_VOCAB_CAP]))
        best.append((score, d, "; ".join(parts)))

    best.sort(key=lambda row: -row[0])
    top_score, top, evidence = best[0]
    runner_up = best[1][0]
    if top_score < _MIN_SCORE or top_score < _MARGIN * runner_up:
        return None
    return DisciplineGuess(discipline=top, score=float(top_score), evidence=evidence)
