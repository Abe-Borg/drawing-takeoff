# drawing-takeoff

Vector-geometry **quantity takeoffs** (linear lengths first) from construction-drawing
PDFs. Measurement runs on exact PyMuPDF **vector geometry** (resolution-independent, whole
sheet, in PDF points); Claude vision is used **surgically** — read the legend once, then
propagate system labels to every same-styled line by exact style match.

> **Status: M1 (geometry + scale proof) ships.** Vendored `core/` infra plus the
> backend-free engine layer — `geometry.py` (the only PyMuPDF module), `models.py`,
> `scale.py`, and a `diagnose` report. Measurement primitives (M2), legend labeling
> (M3), and the GUI/CSV pipeline (M4) are next. See **`IMPLEMENTATION_PLAN.md`** for the
> milestone plan and **`KICKOFF.md`** for the handoff.

## Install & test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # engine deps + pytest
pytest                       # the smoke tests should be green
```

The `[gui]` extra (customtkinter, tkinterdnd2) is added at milestone M4.

Run the M1 go/no-go diagnostic on a vector sheet (cleanliness / instances / scale):

```bash
python -m drawing_takeoff.diagnose path/to/sheet.pdf [--out report.txt]
```

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
  geometry.py          # M1: the ONLY PyMuPDF module (paths + words -> pure models)
  models.py            # M1: backend-free data models (no PyMuPDF import)
  scale.py             # M1: scale-label -> points_per_foot; dimension verifier
  diagnose.py          # M1: go/no-go diagnostic report (pure; runs on models)
  pipeline.py          # M0 stub: extract_takeoff seam (implemented at M4)
  # next per the plan:  measure.py (M2), legend.py (M3), export.py + gui.py (M4)
tests/                 # hermetic harness (sentinel key + SDK fakes) + smoke/scale/diagnose
                       #   + a PyMuPDF-gated geometry test (synthetic vector PDF)
```

## Licensing

Depends on **PyMuPDF (AGPL-3.0)**, so this project is **AGPL-3.0-or-later** (see `LICENSE`).
Keep all PyMuPDF usage isolated to `geometry.py` so the backend stays swappable and the
license boundary stays clean.
