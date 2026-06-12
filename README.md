# drawing-takeoff

Vector-geometry **quantity takeoffs** (lengths and counts) from construction-drawing
PDFs. Measurement runs on exact PyMuPDF **vector geometry** (resolution-independent, whole
sheet, in PDF points); Claude vision is used **surgically** — read the legend once, then
propagate system labels to every same-styled line by exact style match; name one exemplar
crop per repeated symbol, never count by looking.

> **Status: end-to-end POC (M1–M4) + System×Size (M5–M8) + Counts (M9–M12).** Drop in a set
> of vector sheets → a takeoff CSV grouped by system. Geometry + scale (M1), border-aware run
> stitching + per-style footage (M2), legend labeling via one structured Claude call (M3), and
> the `extract_takeoff` pipeline + CSV export + drag-drop GUI (M4) all ship; **`DESIGN_BUCKETING.md`**
> covers the shipped System × Size phase (M5–M8) and **`DESIGN_COUNTS.md`** the counts mode
> (M9–M12): repeated symbols found + counted by congruence clustering, named from exemplar
> crops, totaled as EA per component.

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

# Roll trusted styles up by system AND size, with a marked-up PDF + Excel:
python -m drawing_takeoff.legend  SHEET.pdf --system-size [--out DIR] \
       [--legend LEAD_SHEET.pdf] [--top N] [--no-second-look]

# Count repeated symbols (EA per component) with Claude naming the clusters:
python -m drawing_takeoff.count   SHEET.pdf --label [--out DIR] \
       [--legend LEAD_SHEET.pdf] [--discipline "fire protection"] [--no-second-look]

# The GUI: drag in a set, pick an output mode, confirm the scale, run, save
pip install -e ".[gui]"      # customtkinter + tkinterdnd2
python -m drawing_takeoff.gui
```

The GUI exposes the engine paths and the same knobs as the CLI: an **output
mode** — radio-style, one per run — (*by system* → CSV, *by system × size* →
pipe networks + sizes + second-look re-check, Excel + a marked-up PDF per sheet,
or *Counts* → repeated symbols counted by congruence + named from exemplar
crops, EA per component, Excel + an instance-markup PDF per sheet), an optional
advisory **legend** attachment, and per-mode tuning (top-N networks / max
styles; top clusters / min repeats / max symbol size; second look). A set is
taken end to end and totals roll up across every sheet.

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
python -m drawing_takeoff.count    SHEET.pdf [--markup OUT.pdf]   # M9: repeated-symbol clusters,
                                               # exact counts + every instance boxed on the markup
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
  count.py             # M9/M11: symbol congruence clustering -> exact counts; EA assembly + CLI
  legend.py            # M3/M7/M10: labeling via structured Claude calls (styles / networks / symbols)
  pipeline.py          # M4: extract_takeoff + extract_system_size_takeoff (M5-M8) + extract_count_takeoff (M9-M11)
  export.py            # M4/M8/M11: takeoff CSV + diagnostics + System x Size / Counts Excel workbooks
  gui.py               # M4/M8/M12: drag-drop front-end over the three engine paths (needs [gui])
tests/                 # hermetic harness (sentinel key + SDK fakes) + smoke/scale/measure/
                       #   diagnose + a PyMuPDF-gated geometry test (synthetic vector PDF)
```

## Licensing

Depends on **PyMuPDF (AGPL-3.0)**, so this project is **AGPL-3.0-or-later** (see `LICENSE`).
Keep all PyMuPDF usage isolated to `geometry.py` so the backend stays swappable and the
license boundary stays clean.
