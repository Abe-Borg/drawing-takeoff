# drawing-takeoff

Vector-geometry **quantity takeoffs** (linear lengths first) from construction-drawing
PDFs. Measurement runs on exact PyMuPDF **vector geometry** (resolution-independent, whole
sheet, in PDF points); Claude vision is used **surgically** — read the legend once, then
propagate system labels to every same-styled line by exact style match.

> **Status: end-to-end POC (M1–M4).** Drop in a set of vector sheets → a takeoff CSV grouped
> by system. Geometry + scale (M1), border-aware run stitching + per-style footage (M2),
> legend labeling via one structured Claude call (M3), and the `extract_takeoff` pipeline +
> CSV export + drag-drop GUI (M4) all ship. See **`DESIGN_BUCKETING.md`** for the next
> phase: binding measured footage to system × size buckets.

## Install & test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # engine deps + pytest
pytest                       # the smoke tests should be green
```

## Run a takeoff

The legend/labeling and full pipeline call Claude, so set `ANTHROPIC_API_KEY` first.

```bash
# Name each lineweight's system and roll up trusted styles into a named takeoff:
python -m drawing_takeoff.legend  SHEET.pdf [--legend LEAD_SHEET.pdf] [--discipline "fire protection"]

# The GUI: drag in a set, confirm the scale, run, save the CSV
pip install -e ".[gui]"      # customtkinter + tkinterdnd2
python -m drawing_takeoff.gui
```

Headless, the engine is one call — PDFs in, a `TakeoffResult` out (per-system totals,
flagged styles, per-sheet errors), with `export.write_takeoff_export(...)` for the CSVs:

```python
from drawing_takeoff.pipeline import extract_takeoff
result = extract_takeoff(["FP2.20.pdf", "FP2.21.pdf"], discipline="fire protection")
```

### No-LLM reports (geometry only)

```bash
python -m drawing_takeoff.diagnose SHEET.pdf   # M1 go/no-go: cleanliness / instances / scale
python -m drawing_takeoff.measure  SHEET.pdf   # M2: per-style footage, border-excluded, cross-checked
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
  measure.py           # M2: length primitives, run stitching, per-style footage
  legend.py            # M3: legend labeling via one structured Claude call (style -> system)
  pipeline.py          # M4: extract_takeoff — geometry -> measure -> legend -> totals
  export.py            # M4: takeoff CSV + diagnostics (pure builders + file writer)
  gui.py               # M4: drag-drop front-end over extract_takeoff (needs [gui])
tests/                 # hermetic harness (sentinel key + SDK fakes) + smoke/scale/measure/
                       #   diagnose + a PyMuPDF-gated geometry test (synthetic vector PDF)
```

## Licensing

Depends on **PyMuPDF (AGPL-3.0)**, so this project is **AGPL-3.0-or-later** (see `LICENSE`).
Keep all PyMuPDF usage isolated to `geometry.py` so the backend stays swappable and the
license boundary stays clean.
