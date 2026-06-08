"""Anthropic client factory for the drawing-takeoff.

A small cached wrapper around the Anthropic SDK client. It mirrors the behavior
of the original host app's client factory — read the key from the environment,
cache one client per key — without pulling in any unrelated code. Importing this
module never requires a key; the key is read lazily on first use.
"""
from __future__ import annotations

import os

from anthropic import Anthropic

_cached_client: Anthropic | None = None
_cached_key: str | None = None


def get_client() -> Anthropic:
    """Return a process-wide cached Anthropic client.

    Reads ``ANTHROPIC_API_KEY`` from the environment. Raises ``ValueError`` when
    it is unset so a misconfigured run fails loudly instead of at the API.
    """
    global _cached_client, _cached_key
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    if _cached_client is None or _cached_key != key:
        _cached_client = Anthropic(api_key=key)
        _cached_key = key
    return _cached_client
