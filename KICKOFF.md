# Kickoff — drawing-takeoff POC

This repo is a **proof of concept** for a vector-geometry **quantity takeoff** tool
(linear lengths first) for construction-drawing PDFs. It is a **separate program** from
its sibling `drawing-analyzer`; the only thing borrowed is a small, vendored
infrastructure cluster under `src/drawing_takeoff/` (Anthropic client + `core/`: model
config, pricing, token estimates), already copied and rebranded — so this repo needs
**nothing** from `drawing-analyzer` to build or run.

## How to deploy this

1. Create an empty GitHub repo (e.g. `drawing-takeoff`) with a README + AGPL-3.0 license.
2. Drop the contents of this kit into it and make the first commit.
3. Start a coding session **scoped to this repo only** (not `drawing-analyzer`).
4. Give the agent the instruction below.

## Instruction to give the coding agent (paste this)

> Build the POC described in `IMPLEMENTATION_PLAN.md`, working **only in this repo**. Do
> not modify, push to, or depend on any other repository. Follow the milestones in order
> (M0 → M4). After M0, stop and confirm `pip install -e ".[dev]"` then `pytest` is green
> before starting M1. **M1 is the go/no-go gate and runs on the real sample sheet I will
> provide — do not fabricate or substitute drawing data; ask me for the sheet.** Keep ALL
> PyMuPDF usage isolated to a single module (`geometry.py`) so the AGPL boundary stays
> clean and the backend stays swappable. Commit at the end of each milestone.

## Ground rules

- **Isolation:** everything lands in this repo. `drawing-analyzer` is reference only and is
  not required here.
- **License:** AGPL-3.0-or-later (PyMuPDF). All PyMuPDF usage stays in `geometry.py`.
- **Engine stays headless:** the GUI (M4) is a thin front-end over a clean
  `extract_takeoff(...)` entry point + `progress(done, total, label)` callback. Early
  milestones are script/test-drivable; **M1–M2 use no LLM at all** (pure geometry + math),
  which is what de-risks the concept before any vision call.

## What's already in this repo

- `src/drawing_takeoff/client.py` + `src/drawing_takeoff/core/{api_config,pricing,tokenizer,
  api_key_store,app_paths}.py` — **vendored infra, do not re-derive.** Imports are relative
  (no rewrite was needed); identity strings were rebranded to `drawing_takeoff` /
  `DRAWING_TAKEOFF` / `DrawingTakeoff`. Only `REVIEW_MODEL_DEFAULT` plus the
  effort/thinking/capability helpers are needed for the POC — you may trim the unused
  verification/triage/cache phase config out of `api_config.py` later if you want.
- `tests/` — hermetic harness: sentinel API key (`conftest.py`), generic SDK fakes with a
  `FakeClient` for `client=` injection (`fixtures/fake_anthropic.py`), and a `test_smoke.py`
  that must stay green.
- `pyproject.toml`, `.gitignore`, `LICENSE`, `README.md`, `IMPLEMENTATION_PLAN.md`.

## What to build (net-new, per the plan, in dependency order)

`geometry.py` (the only PyMuPDF module: `get_drawings()` paths + `get_text("words")`),
`models.py`, `scale.py`, `measure.py`, `legend.py`, `pipeline.py` (`extract_takeoff`),
`export.py` (CSV + diagnostics), `gui.py` (adapted thin front-end).

## The one input you must supply

A **single representative vector sheet** — ideally MEP, with pipe/conduit runs, a legend,
and at least one dimension string. It is **not** needed for M0 (scaffolding), but **M1
cannot be validated without it**. Hand it over at the start of M1.
