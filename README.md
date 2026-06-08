# drawing-takeoff

Vector-geometry **quantity takeoffs** (linear lengths first) from construction-drawing
PDFs. Measurement runs on exact PyMuPDF **vector geometry** (resolution-independent, whole
sheet, in PDF points); Claude vision is used **surgically** — read the legend once, then
propagate system labels to every same-styled line by exact style match.

> **Status: POC scaffold.** Only the vendored `core/` infrastructure ships today. See
> **`IMPLEMENTATION_PLAN.md`** for the milestone plan and **`KICKOFF.md`** for the handoff.

## Install & test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # engine deps + pytest
pytest                       # the smoke tests should be green
```

The `[gui]` extra (customtkinter, tkinterdnd2) is added at milestone M4.

## Scale (the simplifying assumption)

Drawings are **true-to-scale** and **labeled** with their scale, so scale is arithmetic, not
calibration: `points_per_foot = 72 × (paper_inches_per_foot)` (e.g. 1/4"=1'-0" → 18 pt/ft),
and `feet = measured_points / points_per_foot`.

## Layout

```
src/drawing_takeoff/
  __init__.py
  client.py            # vendored: Anthropic client factory
  core/                # vendored: api_config, pricing, tokenizer, api_key_store, app_paths
  # built per the plan: geometry.py (ONLY PyMuPDF module), models.py, scale.py,
  #                      measure.py, legend.py, pipeline.py, export.py, gui.py
tests/                 # hermetic harness (sentinel key + SDK fakes) + smoke test
```

## Licensing

Depends on **PyMuPDF (AGPL-3.0)**, so this project is **AGPL-3.0-or-later** (see `LICENSE`).
Keep all PyMuPDF usage isolated to `geometry.py` so the backend stays swappable and the
license boundary stays clean.
