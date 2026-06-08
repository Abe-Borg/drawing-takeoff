"""Engine orchestration entry point (importable stub until milestone M4).

This module is the single, GUI-agnostic seam for the takeoff engine: the GUI
(M4) and any script/test driver call :func:`extract_takeoff`, and everything
upstream (``geometry`` -> ``measure`` -> ``legend``) feeds into it. Keeping the
engine behind one headless, callable entry point is the GUI/engine boundary the
kickoff calls for — the front-end stays a thin shell over this function.

Per the milestone plan this is an *empty, cleanly-importable* stub at M0: the
geometry/measure/legend layers it will orchestrate do not exist yet, so the
signature is fixed now (so the GUI and tests can target it) while the body is
filled in at M4 — progress callback, per-sheet error capture, page-ordered
assembly returning a ``TakeoffResult``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

__all__ = ["extract_takeoff"]


def extract_takeoff(
    pdf_paths: Sequence[str | Path],
    *,
    client=None,
    progress: Callable[[int, int, str], None] | None = None,
) -> "TakeoffResult":  # noqa: F821 - TakeoffResult lands with models.py (M4)
    """Run a linear-length takeoff over ``pdf_paths`` and return the result.

    M0 stub: the geometry, measurement, and legend layers this orchestrates do
    not exist yet, so it raises :class:`NotImplementedError`. The signature is
    locked now because it is the GUI/engine seam — the front-end and tests
    target it before the engine behind it is built (implemented at M4).

    Args:
        pdf_paths: input PDF sheet paths, processed in page order.
        client: optional duck-typed Anthropic client for the legend step
            (M3+), injected in tests via this ``client=`` seam. ``None`` lets
            the engine resolve the vendored default at call time.
        progress: optional ``progress(done, total, label)`` callback so a GUI
            can report per-sheet progress without coupling to the engine.
    """
    raise NotImplementedError(
        "extract_takeoff is a milestone-M0 stub; the engine is implemented at "
        "milestone M4 (see IMPLEMENTATION_PLAN.md)."
    )
