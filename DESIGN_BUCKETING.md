# Design draft: binding measurement to meaning (the bucketing layer)

> **Status: M5–M8 shipped.** M5 (networks), M6 (pipe sizes), M7 (LLM system labeling → System ×
> Size), and M8 (Excel + marked-up PDF output) are implemented and validated on the real sheets
> (§8). Sections 9–10 are the remaining open questions and risks. It extends the shipped POC (M1–M4; see `README.md`) with the
> classification/bucketing layer the POC deliberately stubbed.

## 1. Context — the gap this closes

M1–M4 **measure** exactly (whole-sheet vector geometry, resolution-independent) but **bucket**
only by `StyleKey` — pen color + width + dashes — on the bet that *one pen == one meaning*.
That bet is the thing we're now questioning. It is too coarse in three ways the user named:

- **Blind to size.** 1″ and 2″ sprinkler pipe are the same system, often the same pen. Style
  cannot separate them. (Size was *always* intended — the M3 schema was
  `{style_description, system_name, size, unit}`, outcome `Chilled Water Return, 6": 1,240.5 LF`
  — but it was dropped; `SystemLabel.size` is the unused vestige.)
- **Conflates systems that share a pen**, and inversely **fragments one system drawn with
  several pens** across consultants/sheets.
- **Ignores symbols entirely** (diffusers, heads) — those are *count* takeoffs, not lengths,
  and a length-by-style pipeline has nowhere to put them.

**Input reality (locked with the user):** always **vector** PDFs; true-to-scale; legends are
often **absent, incomplete, or wrong**; **size/system tags are sparse and inconsistent** (many
runs carry no callout at all); different consultants use **different symbols/styles** for the
same component. Assume an inconsistent mess.

**What "always vector" buys us — and why this is tractable:** the two hard problems of a
vision takeoff are *measurement* and *recall* (did we catch every run?). Vectors solve **both**:
the engine sees 100% of the linework every time and measures it exactly. The LLM therefore
**never has to find or measure anything** — it only has to *name* what the engine already
found. The residual problem collapses to **classification**, and a classification error is
*reviewable and bounded* ("this footage landed in the wrong bucket / got flagged"), never a
silent miss of real material.

**The principle the whole design rests on:** *machine measures, LLM classifies, human verifies.*
The LLM **emits no numbers.** It returns a **mapping keyed on identifiers the engine owns**;
code does every join and every sum. Every cell in the final takeoff traces back to exact
geometry.

## 2. The correspondence model — join keys

The engine and the LLM meet on a shared vocabulary of **identifiers**: things the engine
computes deterministically *and* the LLM can perceive and reason about. The LLM speaks meaning;
the engine speaks measurement; they bind on identifiers. Ranked by how much we trust each:

| Join key | Computed by | Carries | Independent of… | Failure mode |
|---|---|---|---|---|
| **Connectivity → network** | engine (graph over run endpoints) | system structure ("follow the line") | color/width/dashes/legend | tee-vs-crossover ambiguity; export gaps break a network |
| **Tag → run/network** | engine (located text, nearest-run) | **system + size** | lineweight | sparse (not every run is tagged); notation varies |
| **Symbol signature** | engine (normalized congruent-geometry hash) | component identity for **counts** | legend | flattened/inconsistent symbol geometry |
| **Style** | engine (`StyleKey`) | weak grouping hint | — | unreliable when the set is messy → demoted to a tiebreaker |
| **Set-of-marks number** | engine (overlay on render) | the *addressing scheme* the LLM points with | — | legibility at high network density |

The reframe: the LLM's deliverable is a **reconstructed legend** — a
`network/symbol → meaning` dictionary built from context (the rendered sheets + the advisory
legend + the engine's structured facts), with the consultant's notation **normalized**. Code
applies that dictionary to its own measurements.

## 3. Architecture / data flow

```
            DETERMINISTIC (engine, exact)                 REASONING (LLM)            HUMAN
 PDF ─► geometry ─► structure ─────────────────────────►  reconstruct legend  ─►  review
        (runs,      • connectivity graph → networks        (mapping keyed on        against
         tags,      • snap tags → runs/networks             network/symbol IDs;      marked-up
         symbols)   • cluster symbols → counts              NO numbers)              overlay
                    • render numbered crops (set-of-marks) ◄─ optional tool calls ─┘   │
                              │                              (measure_network, …)      │
                              ▼                                      │                 ▼
                    aggregate (apply mapping) ◄────────────────────-─┘          corrections
                    • System × Size → LF                                        feed back
                    • Symbol → count
                    • trusted vs flagged
                              ▼
                    outputs: marked-up PDF · System×Size table · Excel
```

Stages 1–2 and "aggregate" are pure/deterministic and unit-testable without a backend (the
existing models discipline). Only "reconstruct legend" calls Claude. The human only ever
confirms **labels**, against a picture.

## 4. The engine → LLM contract (the "sheet facts" object)

A compact, ID-keyed snapshot of one sheet. **Every number in it comes from the engine.** Shapes
(fields, not code):

- `networks: [ { id: "N3", total_lf, bbox, endpoints, member_style_ids, tags_touching: ["T12"],
  size_tags: ["2\""], degree_stats } ]` — connected components of the run graph.
- `tags: [ { id: "T12", text: "2\"Ø", kind: "size"|"system"|"length"|"other", center, nearest_network } ]`
  — located text the engine parsed/classified heuristically (advisory `kind`).
- `symbol_clusters: [ { id: "S7", count, signature_hash, bbox_samples, swatch_ref } ]`.
- `styles: [ { id: "s3", stroke, width, dashes, total_lf } ]` — kept as a *hint*, not the spine.
- **Renders:** high-DPI crops **registered to PDF coordinates**, with networks/symbols
  **numbered** (set-of-marks); plus the advisory legend image if supplied.

Design rules: IDs everywhere; the LLM can cross-reference a number it sees on the overlay to a
record here; the object is small (networks/symbols/tags, not raw paths), so it fits comfortably
in context even for a busy sheet.

## 5. The LLM → engine contract (the reconstructed legend)

One forced tool call (the evolution of today's `record_system_labels`, re-keyed off
networks/symbols and **carrying no quantities**):

- `network_labels: [ { network_id, system, size: { value | "from_tags" | "unknown" },
  confidence, ambiguous, reasoning } ]`
- `symbol_labels: [ { symbol_id, component, confidence, ambiguous, reasoning } ]`
- `structure_edits: [ { op: "merge"|"split", network_ids, at_hint } ]` — optional: let the model
  fix connectivity mistakes it can *see* (a missed tee, a false crossover), expressed as ops on
  engine IDs, not as geometry.

**Optionally agentic.** Instead of one shot, give the model deterministic tools and let it
converge — *every number still comes from a tool, never the model*:
`measure_network(id) → lf`, `tags_near(id, radius) → [...]`, `count_signature(id) → n`,
`highlight(id) → png`. The model decides *what to look at and what it means*; tools own the math.
(One-shot vs. agentic is an open question — §9.)

## 6. Bucketing — how System × Size assembles

The crux of the user's example, made concrete:

1. **System** comes from the **network** label (LLM).
2. **Size** comes from **tags distributed along the network** — because a single network reduces
   (4″ main → 2″ branch), the engine **splits the network's footage into size segments** by each
   run's nearest size tag. Size is a property of *segments*, system is a property of the *network*.
3. **Untagged footage** → an explicit **unsized remainder** under the right system. With tags
   assumed sparse, *this is the common case, not an edge case*: the reliable headline is the
   **per-system total LF**, and the size split is best-effort over whatever tags exist — unsized
   footage is surfaced, never guessed or dropped. (The LLM may *propose* a size from visual
   context or a legend's lineweight→size assertion, but only as an **advisory, flagged**
   suggestion, never silently totaled.)
4. **Symbols** → counts by component, separate table.
5. Each bucket is **trusted** (confident, unambiguous) or **flagged**; only trusted rolls into
   headline totals, mirroring the current trust gate.

Worked shape:

```
System                         Size       Qty     Unit   Basis
Fire-protection sprinkler      (total)  2,041.5   LF     network N3 — the solid headline
Fire-protection sprinkler        2"       420.0   LF     tagged
Fire-protection sprinkler        1"       180.5   LF     tagged
Fire-protection sprinkler      unsized  1,441.0   LF     no callout on run — review / assume
Diffuser (supply)              —             47   EA     symbol S7 ×47
```

## 7. Output & review

- **Marked-up PDF overlay** — the trust artifact: recolor each system/size onto the original
  sheet, circle every flagged item. Lets the estimator *visually confirm we traced the right
  lines* in seconds. Natural because we already own the exact geometry.
- **Clean System × Size table** + **Excel workbook** (Summary / Detail / Review tabs) — lands in
  the estimator's real workflow, unlike three raw CSVs.
- **Review loop:** human confirms/relabels against the overlay; a correction is a label change
  keyed on a network/symbol ID — cheap, and a candidate to remember across sheets in a set.

## 8. Milestone plan (riskiest / cheapest-to-falsify first)

### M5 — Connectivity → networks (NO LLM) — VALIDATED & SHIPPED
**Result on the two real sheets:** the spine holds, but the probe *corrected the model* —
connectivity must be **endpoint-to-segment (tee-aware)**, not endpoint-to-endpoint: a branch tees
into the *middle* of a main, so endpoint-only sharing shatters the system (282 pipe runs stayed
~all singletons). With a **scale-aware ~0.5 ft tolerance** (bridging the breaks pipe picks up at
fittings) the candidate linework collapses sensibly — FP2.20: 282 runs → 31 networks, top-3 = 57%
of LF; FP2.21: 144 runs → 11 networks, top-3 = 92%; at 1 ft FP2.20 collapses to essentially one
connected system (~99%). True crossovers carry no endpoint at the crossing, so a modest tolerance
doesn't merge them.
**Shipped (`measure.py` + `models.py`):** `connect_runs` (grid-indexed, near-linear), `networks`
→ `Network` objects (largest first), `build_networks_report` (+ `measure --networks` CLI), and
hermetic tests for tee-join / gap-bridge / crossover-safety / disjoint.
**Caveat the probe exposed:** the candidate set is the no-LLM `heaviest-dark` heuristic — a rough
proxy that misses branch lineweights and can catch a matchline (FP2.21's two largest "networks"
span 83–85% of the sheet and are auto-flagged). Connectivity stays robust to that; picking the
true pipe styles is M7's job.

### M6 — Pipe sizes (NO LLM) — VALIDATED & SHIPPED
**Probe finding that reshaped it:** sizes here are *not* inch-marked — they're unicode fractions
(`1¼`, `1½`) and bare numbers (`2`, `4`, `6`); `1'-0"` / `12-0` are *lengths*, not sizes. A naive
`NN"` parser finds ~3 tags and concludes "no sizes"; the real notation is dense but inconsistent.
**Result:** with a parser that snaps those forms to a fire-protection nominal set (¾"–8") and
counts a token only when it's adjacent to a pipe run, coverage far beat the "sparse" assumption —
FP2.20 **93% sized** (1½" 437 LF, 1¼" 298, 1" 268, 6" 179, 4" 78, 2" 53; 7% unsized), FP2.21
**96% sized**. The split is a textbook sprinkler breakdown (1"–1½" branches + 2"/4"/6" mains).
**Shipped (`measure.py`):** `parse_pipe_size_in` (unicode/ascii fractions, bare, inch-marked;
rejects off-grid numbers and the `1/8"` scale), `size_tags`, `linear_feet_by_size` (per-run
nearest-tag; **unsized remainder first-class**), `build_size_report` (+ `measure --sizes` CLI),
and hermetic tests. **Caveat:** bare integers (`1`, `6`) are less certain than the unambiguous
`1¼`/`1½` — a detail digit near a pipe could mis-size. The size-set + adjacency filter limits it,
M7's LLM does the final normalization, and the per-network total stays exact regardless.

### M7 — LLM legend reconstruction (LLM enters) — VALIDATED & SHIPPED
**Probe → build:** one Claude call per sheet (Sonnet 4.6, ~2¢) labels the numbered networks and
returns, per id, the system + is_pipe + confidence/ambiguous — keyed on engine ids, **no numbers**.
A single forced tool call sufficed (no agentic loop needed for these sheets). Code then joins
network→system (M7) with run→size (M6) into the **System × Size** table.
**Result on the real sheets:** FP2.20 → one "Fire-protection sprinkler" system, 1,234.6 LF split
1"/1¼"/1½"/2"/4"/6" (+ unsized, + a 170 LF "not labeled" remainder, never dropped). FP2.21 → the
model correctly split the long N1 into a distinct "main/standpipe feed", and the CONFIRM advisory
flagged both page-spanning networks for a human glance — the geometry-proposes / LLM-adjudicates /
human-confirms split the design rests on (it overrode M5's crude `%page` "matchline?" flag using
M6's size evidence).
**Shipped:** `geometry.render_networks_png` (set-of-marks), `legend.network_facts` /
`label_networks` (forced tool call → `SystemLabel` per id) / `system_size_takeoff` /
`build_system_size_report`, a `legend --system-size` CLI, and hermetic tests (fake client).
**Pipe-style selection (shipped):** a first style-classification pass (`label_styles`) now picks
which lineweights are pipe — `pipe_runs_from_style_labels` feeds the union of the **confidently**
pipe styles (`trusted`) into networking, so mains + branches in different pens are captured, not
just the heaviest-dark pen. Measurable-but-ambiguous styles are surfaced as "STYLE TO CONFIRM (not
counted)" rather than silently included (an unsure light-gray style on FP2.20 inflated the total
3.6× until this was gated on `trusted`) or dropped. Two cheap calls per sheet (style pass + network
pass).
**Open:** system-name normalization (FP2.21 split one discipline into two near-identical names);
single-discipline sheets keep the system axis uniform, so cross-system discrimination still wants a
mixed-discipline sheet to prove out.

### M8 — Marked-up PDF + Excel output — SHIPPED (Excel + PDF; GUI relabel deferred)
**Built:** the estimator-facing deliverables, generated by `legend --system-size --out DIR`:
- **`takeoff.xlsx`** (`export.build_takeoff_workbook`, openpyxl) — *Summary* (System × Size with
  per-system totals), *Detail* (one row per network: system, is_pipe, counted, confidence, LF,
  %page, sizes, reasoning), and *Review* (not-counted / confirm notes, incl. the NOT LABELED
  remainder and OTHER DARK lineweights). Fed plain data by `legend.takeoff_tables`, so the writer
  never touches the engine's model types.
- **`takeoff_markup.pdf`** (`geometry.write_marked_up_pdf`) — the networks drawn (vector, colored,
  numbered) onto the original sheet, so the estimator confirms the takeoff traced real pipe at any
  zoom.
Hermetic tests cover the workbook (Summary/Detail/Review cells, size order, totals) and
`takeoff_tables`; the PDF is validated by generating it on the real sheet.
**Deferred:** an in-GUI results table + click-to-relabel review loop (the design's full
human-in-the-loop); the file outputs cover the deliverable for now.

## 9. Open questions to pressure-test

**Settled since this draft:** tee *connection* (endpoint-to-segment) is **shipped in M5** — it's
mandatory, or systems shatter; what stays **deferred** is tee-vs-crossover *disambiguation* (a
near-touch that isn't a real connection), held off by the conservative ~0.5 ft tolerance. Size-tag
density is **locked as sparse/inconsistent** (§1), so the *unsized remainder* is a first-class
output, not an edge case to engineer away.

Still open:

1. **Set-of-marks legibility.** How many networks on a busy sheet? If it's hundreds, numbering is
   illegible and we need hierarchy (label regions → drill in) instead of a flat overlay.
2. **One-shot vs. agentic.** Is the cost/latency of a tool-using loop worth the accuracy over a
   single structured call on the facts object? Likely sheet-dependent.
3. **Cross-sheet identity.** Same system spans sheets — merge by LLM-supplied name, or by network
   continuity across matchlines? (M4 merges by system string today.)
4. **Graph cuts at equipment.** A network can physically join two systems at a pump/tank. Where
   does the engine cut, and is that a `structure_edit` the LLM proposes, or an engine rule?
5. **Reducers.** Where two *tagged* sizes meet with no tag between them, how to assign the
   transition footage (only relevant where tags actually exist).

## 10. Risks

| Risk | If it bites | Design response |
|---|---|---|
| Connectivity over/under-merges | networks span systems or shatter | endpoint-to-segment joins + scale-aware ~0.5 ft tol (M5, validated on real sheets); crossover disambiguation deferred; LLM `structure_edit` ops; overlay makes it visible |
| Candidate pipe set contaminated | a non-pipe style inflates the total, or a pipe lineweight is missed | M7 style pass selects **confident** (trusted) pipe styles; measurable-but-ambiguous styles are flagged "to confirm", not counted; non-pipe summarized — none silently dropped |
| Size mis-attributed (a bare-number tag near the wrong run) | a size segment is wrong | FP-size-set + adjacency filter (M6); unsized remainder first-class; M7 LLM normalizes; the per-network **total** is unaffected |
| Too many networks for set-of-marks | unreadable overlay, poor grounding | hierarchical labeling (region → network); number only candidates |
| LLM tempted to emit quantities | trust collapse | contract forbids numbers; tools own all math; aggregation ignores any number the model returns |
| Symbols flattened inconsistently | counts unreliable | signature-hash + centroid dedup; fall back to vision count on the symbol crop, flagged |
| Cross-consultant notation drift | mislabels across sheets | normalization is the LLM's job; persist human relabels across the set |
```
