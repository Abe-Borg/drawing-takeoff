"""drawing-takeoff — vector-geometry quantity takeoffs from construction-drawing PDFs.

POC scaffold. The public engine entry point (``extract_takeoff``) arrives in
``pipeline.py`` at milestone M4; until then this package ships only the vendored
``core`` infrastructure (Anthropic client, model config, pricing, token
estimates) copied and rebranded from the sibling ``drawing-analyzer`` project.

See ``IMPLEMENTATION_PLAN.md`` for the milestone plan and ``KICKOFF.md`` for the
handoff instructions.
"""
from __future__ import annotations

__version__ = "0.0.1"
