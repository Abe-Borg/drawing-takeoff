"""Hermetic pytest configuration for drawing-takeoff.

Tests never need a real ``ANTHROPIC_API_KEY``: a sentinel is set before
collection so import-time helpers (e.g. the client factory) don't raise. Tests
that need a real network call opt in via ``@pytest.mark.network`` and are skipped
unless a real key is present. Inject the fake client into engine functions via
their ``client=`` parameter (see ``fixtures/fake_anthropic.py``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Repo root on sys.path so ``tests.fixtures.fake_anthropic`` imports; the package
# itself is importable via ``[tool.pytest.ini_options] pythonpath = ["src"]``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SENTINEL_KEY = "test-key-not-real-do-not-use"


def pytest_configure(config: pytest.Config) -> None:
    """Inject a placeholder key so import-time helpers never raise."""
    os.environ.setdefault("ANTHROPIC_API_KEY", _SENTINEL_KEY)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``@pytest.mark.network`` tests unless a real API key is set."""
    real = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if real and real != _SENTINEL_KEY:
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set; skipping network test")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fake_anthropic():
    """Expose the fake-Anthropic helper module as a fixture."""
    from tests.fixtures import fake_anthropic as module

    return module
