# POC: Vector-Geometry Quantity Takeoff (sibling program to drawing-analyzer)

## Context — why this exists

`drawing-analyzer` (the current repo) **reads** drawings: it rasterizes each PDF page
into a 6×6 grid of image tiles and sends them to Claude vision to get a prose *digest*
(sheet number, equipment, schedules, notes). It measures nothing and discards the PDF's
vector geometry (`render.py` is the only PyMuPDF module and it only rasterizes).

We want a **separate program** that does **quantity takeoffs** — starting with **linear
length takeoffs** (pipe / conduit / duct / wall run lengths). Takeoff is a *measurement*
task, not a reading task, so it needs two things the digest pipeline throws away:
exact **geometry** and a **scale**. The approach is **hybrid**:

- **Measurement** runs on whole-sheet **vector geometry** (PyMuPDF `page.get_drawings()`),
  which is exact float coordinates in PDF points (72 pt = 1 inch) and **resolution-independent** —
  so measuring the whole sheet is exact and needs no tiling. This is what dissolves the
  parent's 8% tile-overlap double-count problem: there are no tiles and no seams in the
  measurement path; a run is one continuous polyline in one global coordinate space.
- **Recognition/labeling** (what a line *means* — which system) uses Claude vision **surgically**:
  read the legend **once**, then propagate the system label to every same-styled path by
  exact style match. No per-pipe vision.

**User constraints that simplify the design (lean into them):** vector PDFs only (never
scanned); drawings are **true-to-scale** (no fit-to-page); every sheet is **labeled with
its scale**, consistent across a set. Therefore **scale = read the label + arithmetic, no
calibration**: `points_per_foot = 72 × (paper_inches_per_foot)` → 1/4"=1'-0" gives 18 pt/ft;
`feet = measured_points / points_per_foot`.

**Decisions locked with the user:**
1. **Fresh repo, vendor core** — new repo, copy only the small infra cluster; build takeoff fresh.
2. **Linear lengths** are the POC spine (counts/areas come later).
3. **Reuse the GUI scaffolding** — adapt the parent's customtkinter front-end; the engine stays
   GUI-agnostic behind a clean `extract_takeoff(...)` entry point + progress callback, and the
   early geometry milestones stay script/test-drivable so the concept de-risks before any UI or LLM.

> **Scope note:** This document is a build plan for a coding agent. It states objectives,
> module/function shapes (signatures, no bodies), expected outcomes, and exit criteria.

---

## Repo bootstrap & mechanics

**Approach: fresh repo `drawing-takeoff` (package `drawing_takeoff`), src-layout, AGPL-3.0-or-later.**
Rationale: the POC's first and most important milestone (geometry + scale) needs **none** of the
Claude infra, so a lean repo lets us falsify the whole concept before wiring any LLM or UI. Forking
the parent would drag the digest/synthesis/batch/tiling apparatus we won't use; a shared installable
package is correct DRY but premature for a POC (revisit if the POC graduates).

**Licensing:** the new program also depends on **PyMuPDF (AGPL-3.0)**, so it is likewise
**AGPL-3.0-or-later**. Mirror the parent's discipline: **all PyMuPDF usage isolated to one module**
(`geometry.py`) so the backend stays swappable and the license boundary is clean.

**New-repo mechanics (a human/agent setup step — note: this session's GitHub scope is limited to
`abe-borg/drawing-analyzer`, so it can't create the new repo):**
1. Create an empty GitHub repo, e.g. `abe-borg/drawing-takeoff` (GitHub UI or `gh repo create`).
2. Scaffold per Milestone 0 below; first commit is the vendored core + packaging + green smoke tests.

---

## Module layout (`src/drawing_takeoff/`)

| Module | Responsibility |
|---|---|
| `core/` | **Vendored, unedited** copies of the parent's infra: `client.py`, `api_config.py`, `pricing.py`, `tokenizer.py`, `api_key_store.py`, `app_paths.py`. Anthropic client + cost/config + token estimators. |
| `geometry.py` | **The only PyMuPDF module.** Extract vector paths + text-with-coords from a page; also render the small legend/recognition crops. Net-new. |
| `models.py` | Dependency-free data models (no PyMuPDF import), so `measure`/`pipeline`/`export` reference shapes without the backend. |
| `scale.py` | Scale-label → `points_per_foot`; optional known-dimension sanity check. |
| `measure.py` | Pure geometry math: polyline length (line + Bézier), polygon area (shoelace), segment-stitching into runs, style grouping, instance-signature hashing, text/run association. Unit-testable without PyMuPDF. |
| `legend.py` | The vision step: render legend region, one Claude call (structured/tool-use schema) mapping style→system. Uses vendored `core/client`. |
| `pipeline.py` | Orchestration: `extract_takeoff(pdf_paths, *, client=None, progress=None, …) -> TakeoffResult`. Mirrors parent `pipeline.py` shape (progress callback, per-sheet error capture, page-ordered assembly). |
| `export.py` | `build_takeoff_documents(result, …) -> list[(filename, content)]` (pure) + thin `write_takeoff_export(...)` I/O wrapper. Emits the takeoff **CSV** + a diagnostics dump. |
| `gui.py` | Adapted from parent `gui.py`: drag-drop PDFs, a **scale-confirm field**, run on a worker thread, progress, save CSV. Thin front-end over `extract_takeoff`. |
| `diagnostics.py` | (Optional) reuse parent's leveled-logging pattern. |

---

## Core data models (`models.py`)

- **`GeometryPath`** — `items` (segment list, see geometry design), `stroke_color: tuple|None`,
  `fill_color: tuple|None`, `width: float`, `dashes: str`, `closed: bool`, `bbox: (x0,y0,x1,y1)`,
  `kind: "stroke"|"fill"|"both"`.
- **`StyleKey`** — hashable `(stroke_color, round(width, n), dashes)`; the exact-style grouping key.
- **`TextWord`** — `text: str`, `bbox`, `centroid: (x,y)`.
- **`SheetGeometry`** — `ref` (source PDF + page index, mirror parent `SheetRef`), `page_width_pt`,
  `page_height_pt`, `paths: list[GeometryPath]`, `words: list[TextWord]`, `scale_label: str|None`,
  `points_per_foot: float|None`.
- **`Run`** — a stitched polyline for one style: `style_key`, `polyline: list[(x,y)]`,
  `length_pt: float`, `length_ft: float`, `bbox`.
- **`TakeoffItem`** — `description`, `system`, `quantity: float`, `unit` (e.g. "LF"), `sheet`,
  `location` (bbox/centroid), `style_key`, `scale_used: float`, `confidence: float`,
  `provenance` (run/path ids, run count).
- **`TakeoffResult`** — `items: list[TakeoffItem]`, `per_system_totals`, `sheet_count`,
  `errors: list[str]`, `diagnostics`.

---

## Milestone plan (riskiest / cheapest-to-falsify first)

### M0 — Repo bootstrap (no logic)
**Objective:** installable skeleton with vendored infra and green tests.
**Build:** `pyproject.toml` mirroring the parent (AGPL-3.0; deps `anthropic>=0.97,<0.98`, `pymupdf`,
`tiktoken`, `platformdirs`; `[gui]` = customtkinter+tkinterdnd2; `[dev]` = pytest). Copy the 6 infra
modules into `drawing_takeoff/core/` **unchanged**. Copy test scaffolding: `conftest.py` sentinel
`ANTHROPIC_API_KEY` and `tests/fixtures/fake_anthropic.py` (`FakeMessage`/`FakeUsage`/`FakeTextBlock`,
`client=` injection).
**Outcome / exit:** `pip install -e ".[dev]"` succeeds; `pytest` green on a vendored-core smoke test;
empty `pipeline` imports cleanly.

### M1 — Geometry + scale proof, NO LLM  ← the go/no-go gate
**Objective:** prove, on the user's real sheet, that the vectors are usable and the scale arithmetic holds.
**Build:** `geometry.extract_sheet_geometry(page) -> SheetGeometry` (paths via `get_drawings()`, words
via `get_text("words")`); `scale.points_per_foot_from_label(scale_label) -> float` and
`scale.verify_against_dimension(measured_pt, stated_ft, ppf) -> float` (returns % error); a tiny
script/test entry that dumps diagnostics.
**Outcome:** a diagnostic report answering the three go/no-go questions —
  (a) **cleanliness**: path count grouped by `StyleKey`, the N most common styles, bbox extents;
  (b) **instances**: whether repeated symbols appear as congruent repeated geometry (signature hash counts);
  (c) **scale**: one known dimension measured in points / ppf vs. the stated value
  (e.g. "431.9 pt / 18 = 24.0 ft; sheet says 24'-0' ✓").
**Exit:** scale validated to **<1%** on a known dimension; distinct styles plausibly separate
systems from background. **Needs none of the Claude infra.**

### M2 — Linear measurement primitives, NO LLM
**Objective:** turn geometry into per-style linear footage you can trust.
**Build (`measure.py`):** `polyline_length_pt(points) -> float`; `bezier_length_pt(p0,p1,p2,p3, samples) -> float`;
`stitch_runs(paths_of_one_style, tol) -> list[Run]` (connect endpoints within ε into continuous runs);
`group_by_style(paths) -> dict[StyleKey, list[GeometryPath]]`; `linear_feet_by_style(geometry, ppf) -> dict[StyleKey, float]`.
**Outcome:** a table — `style (blue, 0.5pt, dashed): 1,240.5 ft across 37 runs`.
**Exit:** a **hand-measured run matches the tool within a small tolerance**; per-style totals are stable
across re-runs.

### M3 — Legend labeling (LLM enters)
**Objective:** attach human system names to styles without per-pipe vision.
**Build (`legend.py`):** render the legend region (reuse `geometry`'s rasterizer); one Claude call using
**structured output / tool use** — define a schema like `{style_description, system_name, size, unit}` —
to map observed styles → systems. `label_styles(geometry, styles, *, client) -> dict[StyleKey, SystemLabel]`.
Reuse vendored `core/client.get_client`, `core/pricing.estimate_request_cost`.
**Outcome:** per-style totals carry names — `Chilled Water Return, 6": 1,240.5 LF`.
**Exit:** legend mapping correct for the sample sheet's systems; **ambiguous styles flagged, not guessed**.
(Note: this is the *first* tool-use/structured-output code in either program — the parent is plain-text only.)

### M4 — Structured output, aggregation, CSV + GUI
**Objective:** end-to-end, drag-in-a-set → takeoff CSV, via the adapted GUI.
**Build:** `pipeline.extract_takeoff(...)` (progress callback, per-sheet error capture, page-ordered);
`export.build_takeoff_documents` → `TakeoffItem` CSV + diagnostics file, `write_takeoff_export`;
cross-sheet aggregation (sum by system across sheets); `gui.py` adapted from parent (drag-drop,
scale-confirm field, run, save).
**Outcome:** drop in a set → a takeoff CSV grouped by system with provenance, plus a diagnostics dump.
**Exit:** runs end-to-end on a multi-sheet set; cross-sheet totals reconcile with per-sheet M2/M3 numbers;
**hand-takeoff comparison on the known sheet within tolerance**.

---

## Geometry design specifics (`geometry.py` + `measure.py`)

- **`page.get_drawings()`** returns a list of path dicts. Per path: `items` — a list of segment tuples
  `("l", p1, p2)` line, `("c", p1, p2, p3, p4)` cubic Bézier, `("re", rect)` rectangle, `("qu", quad)`;
  plus `color` (stroke RGB or None), `fill`, `width`, `dashes` (string), `closePath`, `rect` (bbox),
  `type` (`"s"`/`"f"`/`"fs"`). All points are PDF points. **`get_text("words")`** returns
  `(x0,y0,x1,y1, word, block, line, word_no)`.
- **Style grouping:** key each path by `StyleKey(stroke_color, round(width), dashes)`. Construction
  linework for a given system is drawn with one consistent pen → same key. This is the propagation
  mechanism: classify one style, label all its paths.
- **Polyline length:** sum Euclidean distances between consecutive `"l"` endpoints. **Bézier (`"c"`):**
  sample the cubic at N steps and sum chord lengths (PyMuPDF gives no arc length); N≈16 is plenty for
  pipe bends. Most runs are straight; curves are the exception.
- **Polygon area** (for the later area milestone): shoelace on a closed path's vertices in points, × `ppf⁻²`.
- **Segment stitching:** CAD often emits a run as many 2-point segments. Build runs by connecting
  endpoints within ε (a small fraction of a point) so a continuous pipe is one `Run`, not 40 fragments —
  needed for honest run counts and for associating one label per run.
- **Instance detection** (counts, later): normalize each path's geometry to its bbox origin, hash
  `(normalized_items, style)`; identical hashes = repeated symbol instances → exact count + locations.
  Honest caveat: `get_drawings()` returns flattened geometry, so this is signature-hashing, **not** Form/XObject
  introspection — and if the export flattened symbols inconsistently, counts fall back to vision + centroid dedup.
- **Text↔run association:** for a size/system tag, take each word's centroid and attach it to the nearest
  `Run` by perpendicular/centroid proximity — the bridge from "blue 0.5pt dashed run" to its `6" CW` tag.

---

## Reusable infrastructure (vendor verbatim into `core/`)

From `/home/user/drawing-analyzer/src/drawing_analyzer/`:
- `client.py` → `get_client()` (resolves `ANTHROPIC_API_KEY`; singleton).
- `core/api_config.py` → `REVIEW_MODEL_DEFAULT`, `model_supports_effort`, `model_supports_adaptive_thinking`,
  effort/thinking config helpers.
- `core/pricing.py` + `cost.py` → `price_for(model)`, `estimate_request_cost(in_tok, out_tok, *, model, batch)`.
- `core/tokenizer.py` → `estimate_image_tokens(w, h, *, model)`, `count_tokens(text)`.
- `core/api_key_store.py` + `core/app_paths.py` → key load/save (keyring→file), platformdirs paths.

**Patterns to replicate (not copy):**
- **Hermetic tests** — `tests/conftest.py` sets a sentinel API key at collection; `tests/fixtures/fake_anthropic.py`
  provides `FakeMessage`/`FakeUsage`/`FakeTextBlock`; engine functions take a `client=` param (duck-typed, no SDK
  import in signatures). See `tests/test_drawing_digest.py` for the arrange/act/assert shape.
- **Output** — pure `build_*_documents() -> [(filename, content)]` + thin `write_*()` wrapper; folder name =
  slug + timestamp. See `src/drawing_analyzer/export.py` (`build_export_documents`, `write_drawing_export`).
- **GUI/engine seam** — `gui.py` runs the engine on a `threading.Thread`, passes a `progress(done, total, label)`
  callback, marshals results back via `self.after()`. Mirror this; keep `extract_takeoff` callable headless.

**Out of POC scope but reusable later:** `file_upload.py` (generic image upload) and `digest_cache.py`
(generic content-keyed cache — swap its `prompt_version` arg) are takeoff-agnostic; `batch_digest.py`'s
polling loop is reusable if its request-builder is parameterized. None are needed for the POC.

---

## Verification

- **Ground truth:** on the user's known sheet, compare the tool's per-system linear footage against a
  **hand takeoff** (one or two runs measured by hand in Bluebeam/by dimension strings). Target agreement
  within a small tolerance (e.g. ≤1–2%); investigate any larger gap as a stitching/style/scale bug.
- **Scale self-check:** M1's known-dimension check must pass (<1%) before trusting any total.
- **Hermetic unit tests** (no PDF, no network): `measure.py` (lengths/areas/stitching on synthetic point
  sets), `scale.py` (label→ppf table, dimension verifier), `export.py` (CSV shape via `build_*` pure
  functions), `legend.py` (style→system mapping against a `fake_anthropic` tool-use response).
- **PyMuPDF-gated test:** one test that runs `geometry.extract_sheet_geometry` on a **committed fixture PDF**
  (a tiny synthetic vector PDF with a known line of known length), skipped when PyMuPDF is unavailable —
  mirrors the parent's render-test gating.

---

## Risks / unknowns (only a real sheet resolves these)

| Risk | If it bites | Design response |
|---|---|---|
| Hatching/poché explodes path counts | thousands of tiny segments swamp real runs | filter by style + min-length threshold; drop fill-only micro-paths |
| Runs fragmented into 2-point segments | inflated run counts, broken labeling | endpoint-stitching (M2) is mandatory, not optional |
| Outlined text rendered as vectors | text strokes pollute linework totals | exclude paths near `get_text` word bboxes / by style |
| Symbols flattened (no reusable instances) | counts can't come from geometry | out of linear-length scope; note vision+centroid-dedup fallback for the counts milestone |
| Curved runs (Bézier) | length under/over-estimate | sample density N; validate a known curved run |
| Legend absent / not machine-readable | style→system mapping fails | GUI fallback: user maps styles→systems manually |
| Multiple scales / detail viewports on a sheet | wrong ppf for some regions | out of stated scope; later, per-region scale + region tagging |

---

## Required input from the user

**One representative vector sheet** (ideally MEP, with pipe/conduit runs, a legend, and at least one
dimension string) to drive M1–M4. The POC's go/no-go (M1) hinges on what this real file actually contains.
