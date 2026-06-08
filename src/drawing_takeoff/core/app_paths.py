"""App-specific filesystem paths and filenames.

Centralizes the locations Spec Critic uses for persistent state and config —
the API key file and any other app-owned files. Path helpers create
directories on demand so callers can read/write without their own setup
boilerplate.
"""
from __future__ import annotations

import sys
from pathlib import Path

from platformdirs import user_config_dir

API_KEY_FILENAME = "drawing_takeoff_api_key.txt"


def app_config_dir() -> Path:
    d = Path(user_config_dir("DrawingTakeoff", appauthor=False))
    d.mkdir(parents=True, exist_ok=True)
    return d


def executable_dir() -> Path:
    """Directory containing the running source/executable.

    Used as the fallback location for the API key file so the legacy "drop
    a key file next to the .exe" convention keeps working alongside the
    platform config dir.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def api_key_paths() -> list[Path]:
    """Candidate locations to read the API key from, in priority order."""
    return [
        app_config_dir() / API_KEY_FILENAME,
        executable_dir() / API_KEY_FILENAME,
    ]
