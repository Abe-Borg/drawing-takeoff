"""drawing-takeoff — vector-geometry quantity takeoffs from construction-drawing PDFs.

The public engine entry point is ``extract_takeoff`` in ``pipeline.py``: vector
sheets in, a ``TakeoffResult`` out. Measurement runs on exact PyMuPDF vector
geometry (kept isolated to ``geometry.py``); Claude labels line styles by system.
The ``core`` package holds vendored Anthropic infrastructure (client, model
config, pricing, token estimates) from the sibling ``drawing-analyzer`` project.

See ``DESIGN_BUCKETING.md`` for the next planned phase (system × size buckets).
"""
from __future__ import annotations

__version__ = "0.0.1"
