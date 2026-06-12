# Design: the counts takeoff (M9–M12)

> **Status: M9 validated on the real sheets; M10–M12 shipped with this branch.**
> Third output mode — **Counts** (EA by component) — alongside *by system* (LF)
> and *by system × size*. One mode per run (the GUI's segmented button), per the
> user's direction; modes compose later because each is a separate pass over the
> same extracted `SheetGeometry`.

## 1. The principle, unchanged

*Machine counts, LLM names, human verifies.* Recall and arithmetic are the two
things a review can't recover (a silent miss is invisible), so they belong to
the engine: vector exports place every CAD block as the same geometry,
translated/rotated/mirrored, and congruence clustering finds and counts every
placement with exact locations. The LLM never finds or counts anything — it
names **one exemplar crop per distinct symbol** ("Water closet" / "door swing —
not countable"), so LLM cost scales with distinct symbol *types* (~10/sheet),
not instances, and a naming error is bounded, visible on the markup, and
correctable by one relabel keyed on the cluster id. The schemas carry no
numeric fields; aggregation ignores any number the model returns.

No sheet tiling, ever: the engine knows where every instance is, so the model
sees surgical high-DPI crops (`render_symbol_crop_png`, the
`render_network_crop_png` crop-to-target trick) — the "surgical replacement for
blanket sheet tiling" applied to counts. Input reality stays **always vector**
(locked; scanned sheets are out of scope).

## 2. The engine pass (M9, `count.py`, no LLM)

```
suspects    non-degenerate paths with fixture-scale extent (2 pt … 4 ft at sheet
            scale; falls back to 72 pt with no scale — counts are scale-free,
            only this filter and reported sizes use ppf)
  − linework pens   a pen that predominantly draws over-cap geometry (pipe,
            walls) is not a symbol pen; its short fragments are armover stubs
            whose varying lengths would fuse into symbol instances  [probe]
  − exact duplicates  double-drawn parts must not split signatures  [probe]
  − one-off parts   stage 1: a symbol's parts repeat wherever it repeats;
            unique geometry (leaders, unique text outlines) only fuses
            neighboring instances apart  [probe]
instances   co-located parts grouped by bbox proximity (1.5 pt gap, grid-
            indexed); merged groups that outgrow the cap are hatch/pattern
            blobs — excluded, with the count surfaced
signatures  translation-invariant (own-min-relative, 0.1 pt grid, rounded ONCE
            then exact grid arithmetic) and canonical under the 8 right-angle
            rotations/mirrors; per-path serializations sorted so export order
            can't split a cluster
cores       anchor families (instances sharing their most distinctive part)
            propose cores: the POSITIONED, frame-aligned intersection of each
            star-group of similar repeated composites — block parts sit at the
            same canonical offsets everywhere and survive; co-located
            annotation (the pipe-connection tick: 1–3 copies, wandering spot)
            falls out. Similarity is ink-weighted (extent², so a 9 pt disc
            outvotes a 2 pt fragment) with a 0.7 gate, star-shaped around the
            modal composite to stop transitive over-linking  [probe]
atomize     a core that tiles exactly into smaller cores AND whose pieces are
            spatially separable is a co-location (head fused with its size
            label), not a symbol — dropped so constituents get credited; a
            real symbol's parts nest, so position-free single-part atoms can
            never tile it away  [probe]
count       every instance is a bag of co-located symbols: cores are
            constellation-matched into it heaviest-first (anchor fixes the
            translation per frame; every part must sit at its implied position
            ±0.35 pt with the same pen and shape ±0.25 pt), consuming matched
            paths — one fused instance credits the head AND the label.
            Unmatched instances fall back to exact-composite grouping.
```

Outputs: ranked `SymbolCluster`s (id `S0…`, exact count, every instance bbox,
exemplar, drawn size, pens, `variants` = distinct source arrangements
absorbed), singleton arrangements, oversized-blob and one-off-part tallies —
nothing silently dropped. Advisory **fixture-tag cross-check**
(`WC-1`/`LAV2`-shaped text counted from the words layer) rides along: two
independent signals that should roughly agree, the M1 earned-agreement move.

CLI (no key needed): `python -m drawing_takeoff.count SHEET.pdf
[--markup OUT.pdf] [--min-count N] [--max-extent-ft F] [--top N]`.

## 3. M9 probe results (FP2.20 / FP2.21, 1/8" = 1'-0")

The probe **corrected the design four times** — exactly what it was for:

1. **Naive per-path congruence shatters real symbols.** The pendent-head
   symbol is 12–15 flattened paths (polygonized white disc, outline, halves,
   gray dot) *plus* a black pipe-pen armover stub of per-head length and 1–3
   wandering connection ticks. Fixes, in order: linework-pen exclusion (the
   stub), exact-duplicate dedup, one-off-part stage 1, and positioned-core
   extraction (the tick).
2. **Rounding must happen once.** Signatures derived from raw floats vs.
   rounded entries disagreed (±0.1 per coordinate, amplified by the canonical
   frame pick); constellation matching therefore compares geometry with
   tolerance (±0.25/±0.35 pt) instead of hash equality, on a 0.1 pt grid
   snapped up front.
3. **Outlined text exists on real sheets** (size/length annotations as vector
   fills with per-label background masks): they cluster by text content and are
   correctly counted as repeated shapes — *naming* is what excludes them
   ("size label — not countable"), per the division of labor.
4. **Results**: FP2.20 — the head core (8 parts, black 0.24 + fills) counted
   ×47 with the remainder captured by sibling fused/variant clusters; FP2.21 —
   the *identical* core ×51 (S5 on both sheets, same pens). Component-name
   summation in M11 reunifies sibling clusters, and the instance markup makes
   residual misses visible at a glance (boxed vs. unboxed heads). Tick arrays,
   size labels, masks, grilles all surface as their own nameable clusters.
   ~2 min/sheet on a dense E-size enlarged plan.

**Honest caveats** (review surfaces carry them): capture of a symbol can split
across a base core and fused/variant siblings (reunified by name; visible via
`variants` and the markup); two genuinely different symbols sharing one full
core would merge (flagged by a multi-variant review note, second-look crops
show 2–3 variant instances); arbitrary-angle placements land in their own
cluster and merge at the name level; same-scale congruence only (an
enlarged-plan duplicate is the estimator's sheet-selection decision — counts
are reported per sheet).

## 4. Naming (M10, `legend.label_symbols`) — LLM enters

One structured call per sheet, shape-identical to `label_networks`: per-cluster
engine facts (count, drawn size, pens, page spread — all numbers
engine-computed) + one highlighted exemplar crop per cluster + the advisory
legend via the existing transport + discipline. Enforced-schema reply:
`symbol_labels: [{symbol_id, component, countable, confidence, ambiguous,
reasoning}]` — `countable=false` is how door swings, ticks, size labels, grid
bubbles, hatch get excluded *by name*. Flagged clusters get the existing
second-look escalation with wider-margin crops of up to three instances
(context disambiguates: the same oval is a lav on a counter, a urinal on a
wall). Omitted ids default flagged; the trust gate is `SystemLabel.trusted`
unchanged.

## 5. Assembly + outputs (M11) and GUI (M12)

`counts_takeoff`: trusted clusters roll `{component: EA}` (sibling clusters
named alike — including rotated variants and fused captures — sum under one
component); non-countable excluded with a note; flagged/`NOT REVIEWED`/
singletons/oversized to review — never silently dropped. Per-sheet
`CountsSheet` + set-wide `CountsResult` mirror the System×Size shapes, with
the same per-sheet error isolation. Outputs: `counts.xlsx` (Summary /
Detail / Review) + one instance-markup PDF per sheet (every counted instance
boxed in its cluster color, captioned `S3 ×47 Sprinkler head`, `?` when
flagged) — for counts the markup is the trust artifact even more than for
lengths. Tag cross-check appears as advisory review lines. CLI:
`python -m drawing_takeoff.count SHEET.pdf --label [--legend LEAD.pdf]
[--out DIR]`. GUI: third segment "Counts" on the existing mode button, same
legend/discipline knobs, counts knobs (top clusters / min repeats / max symbol
ft / second look), Excel + markup save path.

## 6. Risks

| Risk | If it bites | Design response |
|---|---|---|
| Symbol splits across base + fused/variant clusters | one component, several clusters | M11 sums by component name; `variants` note + markup make splits visible |
| Two symbols share a full core | merged count | multi-variant review note; second-look shows variant instances; human relabel |
| Linework-pen gate eats a symbol pen | symbol parts vanish | per-pen judgment needs ≥5 long paths AND ≥25% long share; tunable; markup shows the miss |
| Outlined text counted as symbols | noise clusters | naming excludes by `countable=false`; extent caps bound it |
| Hatch fields | blob clusters | oversized-blob exclusion with surfaced count |
| Enlarged-plan duplicates across sheets | double count | per-sheet tables; sheet selection is the dedup (v1 decision) |
| LLM tempted to emit counts | trust collapse | schema has no numeric fields; aggregation ignores model numbers |
