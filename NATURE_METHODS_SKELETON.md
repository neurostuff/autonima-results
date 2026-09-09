# Nature Methods skeleton — 3,000 words, 6 display items

Target: **Article**. 150-word unstructured abstract, 3,000 words of body (5,000 at editorial
discretion), ≤6 figures/tables, ~50 references. **Methods sits after the references and is excluded
from the word count** — which is the single most important fact for planning, because it means the
pipeline description, the benchmark construction, and all of `M1. AI Transparency and
Reproducibility` cost nothing against the 3,000.

Format limits from third-party aggregators ([formatting](https://manusights.com/blog/nature-methods-formatting-requirements),
[abstract](https://wordlimit.ai/abstract-limit/nature-methods)); nature.com gates its own guidelines
behind an IdP, so **confirm these before final submission**.

---

## Two papers, written together

Decided 2026-09-04. Both drafted simultaneously; only one goes to Nature Methods.

| paper | venue | role |
|---|---|---|
| **neurometabench** | preprint with DOI, then optionally a data journal | the instrument: nine published meta-analyses with gold study lists *and* per-analysis annotations |
| **autonima evaluation** | **Nature Methods** | the finding: selection at the analysis level is what recovers expert maps |

The split earns back word count where it is scarcest. In a single paper, constructing and
justifying the benchmark would eat several hundred words of a 3,000-word body and a figure panel,
for material that is not the finding. Split out, it becomes **one sentence and a citation**, and
the benchmark paper gets to be as thorough as it should be without a limit.

**The benchmark paper must exist first, or at least simultaneously.** Reviewers will ask where the
benchmark is, and "in a paper we have not submitted yet" is not an answer. A preprint with a Zenodo
DOI is sufficient; it does not need to be accepted anywhere.

## The framing, and why Neurosynth is the exact precedent

Nature Methods does not publish evaluations — the measurement is blunt: **zero
LLM-evidence-synthesis papers across Nature Methods, Nature Machine Intelligence, Nature
Communications, eLife, PNAS and NEJM AI.**

But Neurosynth (Nat Methods, 2011) is the counter-example that shows the way through, because read
closely **it is primarily an evaluation** — the substance is a demonstration that automated
synthesis reproduces what manual meta-analysis produces. It succeeded by presenting that evaluation
as the validation of a new capability rather than as a benchmark exercise. fMRIPrep (2019) did the
same.

So the instruction is not "hide the evaluation". It is **lead with the method and let the
evaluation be its proof**:

> **Automated evidence synthesis has been operating at the wrong unit. Selecting papers is not
> enough — the analysis determines the result, and selecting at that level is both necessary and now
> feasible. Here is a method that does it, and here is what it recovers.**

The existing thesis is already close. The shift is emphasis: §6's decomposition (annotation gain
exceeds screening gain) is the *finding*, not a supporting result. Everything else is evidence for
it.

**What this costs.** §3's cautionary case — currently 1,702 words of outline and one of the
strongest passages — becomes two sentences plus Extended Data. That is the real price of the word
limit, and it is worth paying only if the analysis-unit framing carries.

## Word budget

| block | words | contents |
|---|---|---|
| Abstract | 150 | problem → approach → finding, unreferenced |
| Introduction | ~350 | the unit problem; why coordinate meta-analysis exposes it; benchmark cited, not built |
| **Results** | **~1,900** | six subsections; Results 3–5 get ~350, Results 1/2/6 ~300 |
| Discussion | ~600 | limits of the benchmark; where the gain does and does not appear; what generalises |
| *(Methods)* | *unlimited* | pipeline, benchmark construction, statistics, M1 transparency section |
| **body total** | **~3,000** | the ~100 words the benchmark split returns go to Result 5 |

---

## Section-by-section, with figure allocation

### Introduction (~350 w)

Three moves, no more:

1. **The unit problem.** Evidence synthesis selects papers. But a paper reports many results, and
   for most syntheses only one is the target. Selecting the paper and taking everything in it is
   the standard failure, and it is invisible to every paper-level metric.
2. **Why neuroimaging exposes it.** Coordinate-based meta-analysis has a machine-readable result
   unit (the analysis, with its peaks) and published gold standards. So the error is measurable
   here in a way it is not elsewhere.
3. **What we built.** A pipeline that selects at both levels, evaluated against **neurometabench**
   (cite the companion preprint — do not rebuild it here). One clause that it is the best available
   reference rather than ground truth; the caveat returns in Discussion, its construction does not.

*(One paragraph of the §"Framing" caveat, moved here from where the outline currently has it.)*

### Result 1 — The pipeline, the benchmark, and the unit that matters — **Figure 1** (~300 w)

Schematic. Establishes what is being measured before any number appears.

Panels: **a** pipeline stages with what each consumes and emits; **b** the unit distinction — one
paper, k analyses, one target, and what a paper-level selector does with that. Panel b is the
argument of the paper in a picture and should be drawn first.

The benchmark panel that would otherwise sit here belongs in the companion paper; a one-line
inset naming the nine meta-analyses is enough. That frees the third panel for the unit
distinction to be drawn properly rather than crammed.

### Result 2 — Screening raises precision at almost no cost to recall — **Figure 2** (~300 w)

§1. Panel **a** is cumulative retention of the gold standard across search → abstract screening →
full-text retrieval → full-text screening. Panel **b** is precision over the same funnel.

The claim, now quantified: **abstract screening costs a median 1.4 points** of the gold standard,
while precision roughly doubles or triples. Marginal loss by stage, in percentage points of gold:

| stage | median | max | projects losing >5pp |
|---|---|---|---|
| abstract screening | **1.4** | 8.8 | 2/9 |
| full-text retrieval | 3.4 | 11.9 | 3/9 |
| full-text screening | 4.0 | 14.6 | 3/9 |

**Retrieval is shown as its own stage, not folded into screening.** They fail for different
reasons — no obtainable full text is a supply problem, a rejection is a judgement — and in several
projects retrieval is the larger loss. Merging them would blame screening for the pipeline's
biggest cost.

Seven of nine projects end within 11 points of where their search started. The two exceptions,
executive function (74% → 39%) and problem solving (88% → 58%), lose most of it at retrieval and
full-text screening rather than at the abstract stage, and both should be named rather than
averaged away.

**Dotted overlay: the fixed-pool arm** (`vN-allstudies`), for the three projects that have one.
Same screening and annotation over a pool assembled without search-driven narrowing, so the gap
between solid and dotted is what the *pool* contributes, separated from what screening contributes.

Precision (panel b) is higher for the fixed pool at **every stage, in all three projects**:

| project | canonical, full text | fixed pool |
|---|---|---|
| social | 0.406 | **0.685** |
| dementia | 0.401 | 0.427 |
| emotion regulation | 0.239 | **0.361** |

So the canonical arm's lower precision is substantially a property of what the broad search
returns, not of the screener's judgement — which is the direct answer to "is precision limited by
the mixed pool?" for these three.

**The two sentences that must appear in the body** (the quantification and its control live in
**Supplementary S3**): across the three projects the pool accounts for **+0.142 of precision at
full-text screening** — a mean of 0.349 on the search pool against 0.491 on the fixed pool, so
29% of the precision achievable at that stage is a corpus difference rather than a screening
failure. It is a genuine pool effect and not a threshold trade, because **recall is unchanged**
(mean Δ +0.000 at abstract screening, +0.007 at full text): a precision gain bought by discarding
borderline true positives would show up as a recall loss and does not.

Without those two sentences a reader takes 0.35 as the pipeline's precision ceiling when 0.49 is
reachable on a matched pool, so this is not an optional aside. S3's one visible exception is
emotion regulation's search-stage recall (−0.261), which is the same fact as ER's fixed pool
holding only 56 of 88 gold studies, described below — the two arms are different corpora at that
stage by construction, and every later stage is ~0.

**Retention (panel a) does not follow the same pattern, and the exception is the interesting
part:**

| project | search: canonical → fixed | end: canonical → fixed |
|---|---|---|
| dementia | 96% → **100%** | 85% → **91%** |
| social | 95% → **100%** | 93% → **97%** |
| emotion regulation | 90% → **64%** | 83% → **60%** |

For dementia and social the fixed pool contains every gold study, so it dominates on both axes.
**For emotion regulation it contains only 56 of 88** — the fixed pool *misses gold the search
found*. That single case is worth naming rather than averaging: a hand-assembled pool is not
automatically a superset of a search, and ER's is the project where the benchmark's own scope is
hardest to reproduce.

**Do not plot per-stage recall from `screening_metrics_top_v_stage_progression.csv`.** It holds
*conditional retention* — the share entering a stage that survives it — so its denominator moves
at every stage and the numbers rise across the funnel, which cumulative recall cannot do. Chaining
those retentions does not recover the truth either, since retrieval loss is absent from that file
(executive function chains to 0.645 against an actual 0.392). Use
`reports/gold_survival_by_stage.csv`, which counts gold PMIDs against a fixed denominator.

### Result 3 — Recovering the analyses, then selecting among them — **Figure 3** (~350 w)

Two analysis-level operations in pipeline order, combined into one display item to stay inside six.
Panel **a** is recovery, panel **b** is selection.

**a. Parsing (§ new).** Against a table-only baseline that takes coordinate tables as parsed
without an LLM reading them:

| | expert analyses recovered |
|---|---|
| tables only | **31%** |
| LLM parsing | **90%** |

**+60 points pooled over 1,358 gold analyses, and the LLM wins in all nine projects** — margins
from +36 (VBM PTSD) to +82 (dementia). This is the strongest single margin in the paper and it was
not previously a headline result. It also does necessary work for the argument: the analyses have
to be *recovered* before selecting among them means anything, so this is the precondition Result 5
depends on. Worst case is emotion regulation at 63%, which is also the project with the loosest
target — worth naming rather than averaging away.

**b. Annotation (§5).** Each project is an operating point in precision–recall space, with a
connector down to its own **prevalence**. Exhausted-manual basis, `mode_id = combined`.

**Why the baseline sits at the same recall.** A random selector that picks each analysis with
probability *p* achieves recall = *p* and precision = prevalence, **independent of *p*** — the
selected set has the pool's composition whatever its size. So the no-skill baseline is a
*horizontal line spanning all recall*, not a point, and there is no recall value at which random
"lands". Pinning it to each project's own recall is the like-for-like comparison: at the recall we
achieved, chance would have scored prevalence. The figure draws it as a short horizontal rule
rather than a marker so the encoding says "level", and the caption must state this or a reader will
reasonably ask why random is at recall 0.8.

No pooled baseline line is drawn: prevalence ranges 0.104 to 0.345 across these projects, a 3.3×
spread, so a single mean line would read as *the* baseline and misplace most of them.

**Two presentational choices to state in the caption.** The recall axis is clipped to 0.62–1.02;
every project sits above 0.70 recall, so the full 0–1 range left 60% of the panel empty and
crowded the points into a band where their spans overlapped. And points are labelled directly
rather than by legend — panel **a** already names every project on its y-axis, so colour there is
redundant, while panel **b** would otherwise depend on colour alone with no key. Labels sit beside
the markers because projects cluster in recall (three pairs within 0.02) but spread in precision.

Two reasons this encoding rather than a precision-vs-recall dumbbell. A connector implies a
before/after, and precision and recall are two coordinates of one operating point, not a
progression — worse, it made panel **b** mimic panel **a**, where the connector genuinely does run
baseline → outcome. Here the connector *is* that relationship: random → achieved.

It also makes the absolute numbers interpretable, and **it reorders them**:

| | precision | prevalence | lift |
|---|---|---|---|
| vbm_of_substance_use | 0.584 | 0.104 | **5.6×** |
| vbm_of_ptsd | 1.000 | 0.250 | 4.0× |
| executive_function | 0.545 | 0.158 | 3.5× |
| … | | | |
| social | 0.618 | 0.345 | **1.8×** |

Mean **3.3×** over random. Social's 0.618 is the third-highest raw precision but the *lowest* lift,
because its prevalence is the highest in the set; vbm_of_substance_use turns a similar 0.584 into
5.6×. Reporting raw precision alone would have ranked these backwards — a reader has no way to
judge 0.54 without knowing what chance looks like.

dementia is excluded from **b** only (its gold analyses pool several studies each, so per-paper
extraction has no clean mapping to its units); it parses perfectly at 100% and stays in **a**. The
asymmetry is marked with an asterisk rather than hidden.

State the aggregation dependence in one sentence, because it reverses the conclusion: matched-only
gives precision 0.863 / recall 0.810 (precision-heavy), exhausted-manual gives 0.540 / 0.809
(recall-heavy). Naming the variant is not optional.

### Result 4 — The whole pipeline beats a search-only synthesis — **Figure 4** (~350 w)

§7, the headline. 35 columns, autonima against the strongest baseline a competent practitioner
could have built per column.

**Panel a encodes N.** Circle area is the number of analyses the pipeline pooled for that column
(8 to 1699, so the scale is logarithmic), from `reports/analysis_counts.csv`. It answers the
obvious "are the wins just the well-powered columns?" objection backwards: the pipeline wins 30 of
35, and in **23 of those it pools fewer analyses than the baseline it beats** — median baseline
2.5× larger, 11× for emotion regulation's `increase` (56 analyses at *R²* 0.50 against 627 at
0.15). Selection beating volume is the thesis, so it belongs in the headline figure.

```
32 columns (see the exclusion note below; was 35)
best AVAILABLE   0.574 vs 0.476   Δ +0.099   95% CI [+0.038, +0.181]   sign test p = 1.9e-05
STRONGEST        0.574 vs 0.482   Δ +0.093   95% CI [+0.032, +0.176]   p = 1.1e-04
                 ahead in 28/32 (best available), 27/32 (strongest); median +0.061
```

> **The denominator is 32, not 35 — decided 2026-09-09.** `vbm_of_substance_use` cannabis,
> opioids and stimulants are dropped from every analysis and every plot, because
> **the source paper reports no significant result for them**: "Drug-specific meta-analyses for
> cannabis, opioids, and stimulants failed to yield significant clusters" (Hill-Bowen et al. 2022,
> PMID 36115222). There is no reference finding to recover, so any agreement score is measuring
> agreement with a null. Encoded once in `scripts/benchmark_exclusions.py` and imported by every
> compiler and figure, so the denominator cannot be 32 in one table and 35 in another.
>
> **State this in Methods as a scope decision**, with the quote. It is not a metric choice and it
> does not depend on which metric a figure reports.
>
> Dropping them *lowered* the headline slightly — Δ +0.110 → +0.099, ahead 31/35 → 28/32 — because
> under unthresholded r² all three scored as wins (cannabis +0.135, opioids +0.192, stimulants
> +0.356). Those wins were two null maps agreeing about where the literature puts coordinates, so
> removing them makes the number smaller and honest. Worth one clause: the exclusion costs us,
> which is the best evidence it was not chosen to flatter the result.

CI is cluster-bootstrapped over projects, not columns — a project's columns share a corpus, a
search and a screening run. Say so in one clause; it pre-empts the obvious reviewer objection and
costs nothing, since both intervals exclude zero.

**A second candidate display item, new 2026-09-08:** `figure_er_surface_contrasts`, built by
`scripts/make_er_surface_figure.py`. Three emotion-regulation contrasts on the cortical surface,
expert above and pipeline below, against the single fixed-pool arm that cannot tell them apart —
the analysis-unit argument rendered rather than scattered. *R²* per contrast is computed from the
maps actually drawn, not read from `cross_project_best_baseline.csv`, whose "best available"
baseline for two of these columns is a targeted search rather than the map shown; against the
fixed-pool arm the margins are larger (reappraisal 0.81 vs 0.33, against 0.81 vs 0.51 for the
targeted baseline), so the caption must say which baseline it is.

This takes the sixth slot, freed by moving the cost figure to Supplementary S1 (decided
2026-09-09). Two caveats for the caption: the cortical surface cannot show amygdala or other subcortical structure, which
matters for emotion regulation specifically; and the shared colour ceiling is set to z = 5, which
saturates the baseline (peak z 23) — conservative, since the claim is that the baseline is
undifferentiated rather than weak.

**Add the brain maps here.** Every number in this section is a similarity between two maps and the
reader currently never sees one — a conspicuous gap in a neuroimaging paper, and the panel most
likely to make the result *feel* true rather than merely reported. Panel **c**: axial slices for
five columns, three arms each (search-only baseline / full pipeline / expert meta-analysis).
Built by `scripts/make_brain_map_figure.py`.

Emotion regulation `decrease` is the case that carries it: the baseline is diffuse blue across
frontal and parietal cortex, the pipeline resolves focal bilateral clusters, and the expert map
matches the pipeline closely. Δ*R²* +0.491.

> **⚠ `decrease` cannot be used as an exemplar — resolved 2026-09-08.** Its expert map correlates
> **r = 0.972** with `reappraisal`'s. They are effectively the same map: neurometabench's
> `Decrease.txt` holds the reappraisal union rather than the decrease contrast. The defect is
> recorded in `projects/emotion_regulation_2022/nmb_mappings.json` ("the decrease and maintain
> columns are not interpretable until the gold is rebuilt") and was confirmed independently by
> direct gold-vs-gold comparison. It is also visible in the run's own similarity matrix, where
> auto `reappraisal` scores 0.863 against gold `decrease` — nearly its 0.884 diagonal.
>
> This matters more than a single exemplar: `decrease` carries the **largest single advantage in
> the whole benchmark** (Δ*R²* +0.491), so it is the most tempting column in the set and it is
> currently doing rhetorical work in this section. Two consequences to settle before submission:
>
> 1. **Pick a different exemplar** — `reappraisal` (Δ +0.305) makes the same point on gold that is
>    not duplicated. `scripts/make_er_surface_figure.py` already excludes `decrease` by default.
> 2. **Decide whether `decrease` and `maintain` stay in the 35-column denominator at all.** They
>    are currently counted in every pooled statistic in Results 4 and 5. If the gold is rebuilt
>    they should be recomputed; if it is not, dropping them changes Δ and the sign test and that
>    recomputation has not been done. `maintain` is the weaker claim of the two — its gold
>    correlates 0.03–0.09 with every other ER contrast, which is equally consistent with a genuine
>    contrast and with the missing-Look-experiments defect the note describes, and cannot be
>    settled from the maps alone.

**Exemplars are chosen on end-to-end margin, which is the right criterion for this panel.** Result 4
is the end-to-end claim; Result 5 is the analysis-selection claim. Social is the case that separates
them and is worth a sentence: its annotation is the weakest in the set (F1 0.660, lift 1.8× over
prevalence, median map gain −0.020) while its end-to-end performance is above average (mean dice
0.543 against 0.423 pooled), because 93% gold retention and 94% analysis recovery do the work
instead. That is a reason not to reuse social as a *Result 5* exemplar, not a reason to exclude it
from a figure about end-to-end maps.

> **What the dice threshold actually is — checked 2026-09-09.** `z_corr > 1.96` selects exactly
> the voxels with FDR-corrected *p* ≤ 0.05 on these maps (voxel-identical on five columns across
> three projects; `z_corr` is a strictly monotone transform of the corrected *p*, Spearman
> −1.0000). So the project's 1.96 is the *q* ≤ 0.05 boundary, not an arbitrary display level —
> worth one clause in Methods because it looks arbitrary otherwise.
>
> Two consequences. **(a)** The equivalence uses the two-tailed *z* convention while MKDA is
> one-tailed, for which *q* ≤ 0.05 is *z* ≈ 1.645. NiMARE's own `tail-positive` label mask uses
> that boundary and flags more voxels (22,096 vs 18,427 for `reappraisal`), so 1.96 is
> **conservative** — the safe direction for a positive claim, but say so rather than let a
> reviewer find it. **(b)** This figure renders at *z* > 2.3 while reporting dice computed at
> 1.96, so the number does not describe the picture it is printed beside — 2.3 is roughly
> *q* ≤ 0.025. Either render at 1.96 or recompute dice at 2.3; the surface figure ties the two
> together for exactly this reason and verifies the result against
> `baseline_vs_autonima.csv`.

**This figure reports dice, not r².** Deliberate, and the one place a different metric is
correct: the panels are rendered *thresholded* at |z| > 2.3, and dice measures overlap of
thresholded maps, so it describes what the reader can actually see. r² measures unthresholded
correlation, which the rendered slices do not show. `make_brain_map_figure.py --metric` defaults to
dice for this reason and re-reads the per-project tables rather than the pooled one.

**On cherry-picking.** Selecting exemplars by margin is cherry-picking and should be stated as
such, then defused two ways. The figure runs `--mode contrast`, which pairs the three largest
margins with the three smallest so near-ties are shown beside wins.

**Say in the caption that the bottom rows are small wins, not losses**, and why — otherwise six
positive margins read as selection. Under dice only **1 of 35** columns has a negative margin
(vbm_of_ptsd, −0.214), and it is excluded because its maps render as empty brains at the display
threshold; four more sit at exactly 0.000 and are excluded for the same reason. So among columns
that can be *shown*, none lose. That is itself a reason r² is primary for Figures 4 and 5: r² has
five losses across the same 35 columns, so it discriminates where dice compresses losses into
ties and exclusions. And the supplement carries
**all 35 columns** (`--mode all`), so the reader can check the selection. Figures 4a/4b already
report the full distribution including the five columns where the pipeline loses, so the exemplars
illustrate rather than stand in for the evidence.

**Two selection rules worth stating in the caption**, because both are choices a reviewer could
otherwise read as convenient:

- **One row per project**, not the top five columns overall. Unfiltered, the top four are all
  emotion regulation, which shows the same contrast four times and says nothing about breadth.
- **Columns whose maps render blank are excluded from the main figure.** VBM PTSD has a healthy
  +0.209 margin but only 21 gold studies, so 401 / 32 / 101 voxels survive FDR correction against
  ~5,000–48,000 in the legible rows — three empty brains that occupy a row without informing.
  Those columns still appear in the supplement.

**Correctness note.** Map paths are resolved the way `compare_baselines_to_benchmark.py` resolves
them, and every triplet is verified by recomputing pipeline-vs-expert *R²* and comparing against
`cross_project_best_baseline.csv` (35/35 pass). A wrong path would produce a plausible figure that
does not match the text, which is the failure mode worth engineering against.

### The metric/map rule — decided 2026-09-09

**r² compares unthresholded maps. Dice compares FDR-corrected thresholded maps.** One sentence in
Methods; it removes a whole class of reviewer objection.

The reasoning is that each metric presupposes a map. Dice is an overlap of suprathreshold
volumes, so it needs a map whose threshold means something — the FDR-corrected z at the corrected
*q* ≤ 0.05 boundary, which on these maps is exactly *z* > 1.96 (verified voxel-identical against
the corrected *p* map on five columns). r² is a correlation over all voxels, so it needs the map
that still *has* all voxels; correlating a corrected map means correlating an image whose
sub-threshold structure — most of what the correlation measures — has been zeroed.

**Before this, each script had one half right and one half wrong:**

| script | feeds | dice was on | r²/pearson was on |
|---|---|---|---|
| `compare_baselines_to_benchmark.py` | §7, Figure 4 | corrected ✓ | corrected ✗ |
| `compare_meta_to_benchmark.py` | §6, old Figure 5 | uncorrected ✗ | uncorrected ✓ |

Both are fixed and each now loads both maps. Two knock-on corrections:

**Figure 4's headline moved, slightly upward.** Δ*R²* +0.101 → **+0.110**, columns ahead 30 →
**31 of 35**, sign test *P* 2.2e-05 → **3.5e-06**, 95% CI [+0.043, +0.186]. Absolute *R²* rises
(0.495 → 0.559) because unthresholded maps simply correlate better; the *margin* is what matters
and it is unchanged in character.

**The four "degenerate" substance-use columns were an artefact of the wrong map, not a real
limitation.** An earlier note here called them uninformative because nothing survived FDR, so
dice was identically zero and r² was computed over a near-empty image. On the unthresholded map
they carry real signal: cannabis 0.021 → **0.235**, opioids 0.085 → **0.375**, stimulants 0.269 →
**0.585**, nicotine 0.187 → **0.361**. Cannabis flips from a loss to a win, which is the single
sign change across all 35 columns. **Retract the "report them as uninformative" recommendation.**
What remains true is narrower: *dice* is unusable for them, because their corrected maps are
genuinely empty — and for cannabis the *expert* map peaks at *z* = 0.496, so no pipeline output
could score above zero on dice there. That is a fact about dice at small N, which §8d already
records, not about the columns.

Two display consequences, both now enforced in code rather than left to convention:

- `make_brain_map_figure.py` rendered at *z* > 2.3 while printing dice computed at 1.96 — a number
  describing a map the reader could not see. It now renders at 1.96.
- `make_er_surface_figure.py` ties its display threshold to its dice threshold as one setting, and
  `--metric r2` switches both map paths to the raw z rather than correlating the rendered map.

### Metric consistency — resolved 2026-09-04: r² throughout

Both figures report **r²**. Note this was **already decided** in `PAPER_OUTLINE.md` §8d — "Dice is
unusable at small N — 8 of 12 sub-annotation comparisons gave exactly 0.000 — so R² on
unthresholded maps should be the reported metric, with Dice at most secondary" — and again in §8a,
"r² sits consistently above dice… and the reason r² is the primary metric". Figure 5 used dice only
because of the bug below, not by choice. The re-derivation below adds the mechanism and one new
case; it does not overturn anything.

**Dice is degenerate on this corpus, so Figure 4 cannot use it.** Four `vbm_of_substance_use`
columns (nicotine, opioids, stimulants, cannabis) score **dice 0.000 for every arm** — no
suprathreshold overlap at all — so it cannot rank them, while r² separates them cleanly (nicotine
0.187 vs 0.066). Dice also reverses the sign on vbm_of_ptsd, whose map is *correct but sparse*: 7
studies and 72 peaks against the baseline's 29 and 386, scoring dice 0.111 vs 0.325 while scoring
r² 0.456 vs 0.247. A thresholded overlap measure punishes missing extent; a correlation rewards
matching shape. Corpus-wide, columns where the metrics disagree or tie have a median points ratio
of **0.17** against **0.28** where they agree — systematic, not one odd column.

**Figure 5 could not use r² until a bug was fixed.** `annotation_value.csv` populated
`pearson_annotated` for only 14 of 35 rows, because that lookup used the manual column name where
the dice lookup correctly used the mapped annotation name — so it returned blank for every project
whose annotation names differ from its gold column names. One word in
`scripts/annotation_value.py`; now 35/35.

With that fixed, r² is both available everywhere and the stronger measure for Result 5:

| metric | median gain | mean | improve |
|---|---|---|---|
| **r²** | **+0.0735** | **+0.0995** | 25/35 |
| dice | +0.0372 | +0.0650 | 26/35 |

Per-project ordering is nearly identical (only decision_making flips sign, +0.025 → −0.006), so
the choice does not drive the story — and reporting both results in the same units removes the
awkwardness of Result 5's headline looking three times weaker than Result 4's for no real reason.

`compile_best_baselines.py --metric {r2,dice,pearson_r}` keeps the robustness check runnable:
dice gives Δ +0.107 with the same 30 of 35 columns.

One argument against r² that turns out **not** to apply: r² discards sign, so an anti-correlated
map would score as well as a correlated one. Checked — 0 of 104 comparisons have negative pearson
r. Do not use it as the justification.

### Result 5 — The gain comes from analysis selection, not paper selection — **Figure 5** (~350 w)

§6. **This is the paper.** Holding the study pool fixed, annotation still improves the map — so the
advantage is not explained by retrieving better papers.

> **⚠ Two changes here, 2026-09-08. Read both before quoting any number in this section.**

**1. The comparison is now against a size-matched null, not against `all_analyses`.** The old
comparison had a confound worth naming because a reviewer will find it: annotation both *chooses*
analyses and *shrinks* the set, and a smaller CBMA is not simply a worse one — changing N changes
the density map's scale and sparsity. So part of a gain over `all_analyses` could have come from
using fewer analyses, whatever they were.

`scripts/bootstrap_annotation_null.py` removes it. For each column it draws 500 random subsets of
the annotated set's size from the same pool, runs the same MKDA + FDR, and scores each against the
same expert map. The annotated map's position in that distribution is the effect of selecting
*well*, with selecting *fewer* held constant. Figure 5a shows every column against its own null;
5b summarises per project.

**Final numbers, 32 columns, r² on unthresholded maps per the metric/map rule: median Δ*R²*
+0.211 over a same-size random selection, 29 of 32 columns clear their own null at *P* < 0.05,
and 24 sit outside all 500 draws** (*P* at the 1/501 floor). Mean Δ +0.222. The dice arm from the
same draws is reported as secondary: median Δdice +0.170, 24 of 32 at *P* < 0.05.

**Report the three failures; they are the evidence the null discriminates**, and they now tell one
story rather than three:

| column | k / pool | share of pool | observed | null | *P* |
|---|---|---|---|---|---|
| `decision_making` adm | 85 / 274 | 31% | 0.272 | 0.251 | 0.26 |
| `social` all_merged | 557 / 720 | 77% | 0.821 | 0.808 | 0.13 |
| `social` others_merged | 449 / 720 | 62% | 0.727 | 0.702 | 0.06 |

**All three select a large share of the pool, and that is the mechanism.** As the annotated set
approaches the pool, the null approaches the observed value and there is nothing left to select —
so a column can fail here without annotation having done anything wrong. This is §6's "the gain
scales with how selective the target is" appearing as a mechanism rather than a correlation, and
it is a much better sentence than "three columns were not significant".

The middle row generalises and is worth one sentence in the text: **pooled/global columns have
little headroom by construction**, because the annotated set approaches the pool and the null
approaches the observed value. That is the same effect §6 records as "the gain scales with how
selective the target is", now visible as a mechanism rather than a correlation.

**2. The old numbers were computed on a different brain map than Figure 4's.** Not staleness —
that was an earlier misdiagnosis, corrected here. `annotation_value.csv` comes from
`compare_meta_to_benchmark.py`, which defaults to the **uncorrected** `z.nii.gz`; Figure 4's table
comes from `compare_baselines_to_benchmark.py`, which defaults to the **FDR-corrected**
`z_corr-FDR_method-indep.nii.gz`. So Figures 4 and 5 were reporting *R²* on different maps.

Confirmed by reproducing the committed values from `z.nii.gz` to four decimals (alcohol 0.8333,
nicotine 0.4372, cannabis 0.5388), which a stale file could not do. Recomputing on the corrected
map moves **26 of 35 columns, almost all downward**, because correction zeroes the sub-threshold
voxels that carried much of the correlation. Median Δ*R²* +0.074 → **+0.064**, still 25/35.

The bootstrap uses the corrected map, so Figure 5 is now consistent with Figure 4. **State the
convention once in Methods** — this is the kind of split a reviewer recomputes and finds.

Same units as Result 4, but still a different comparison — Result 4 is pipeline vs search-only
baseline, Result 5 is analysis selection against chance at matched N. Quote its own number.

Two panels. An earlier draft was a single panel paired with a slope plot of baseline vs annotated
dice across all 35 columns — 35 crossing lines in 89mm, unreadable, and carrying nothing the gain
distribution does not already show except absolute dice levels, which Figure 4 supplies. The
per-column forest in 5a is what that slope plot was reaching for and could not deliver.

**Two candidate mechanisms were tested and belong in the text, not in a panel:**

| candidate | result | verdict |
|---|---|---|
| gain vs prevalence (target specificity) | Spearman −0.233, n = 9 projects | right direction, no power |
| gain vs baseline dice (headroom) | Pearson −0.353, t = −2.17, n = 35 | marginal, and deflationary |

The first is the mechanism the paper argues for — narrower target, more to gain from selecting —
and it points the right way, but nine projects cannot support it. The second is the ceiling-effect
objection a reviewer will raise unprompted, so **report it rather than wait for it**: gain is
somewhat larger where the baseline was worse, though the quartiles are not monotone (+0.116,
+0.035, +0.084, +0.007). Neither is strong enough to plot without implying more than it supports.

That leaves the per-project ordering as the qualitative mechanism: the projects requiring a
specific population contrast gain most, and the two that lose — executive function and social — are
the most diffuse targets in the set. Say it as an observed ordering, not a tested relationship.

### Result 6 — Cost and scale — **Supplementary S1** (~200 w)

> **Moved to the supplement 2026-09-09.** Nature allows six display items and the
> emotion-regulation surface figure is a stronger use of the slot than a cost bar chart. Cost
> becomes a short paragraph in the text pointing at S1, which frees Result 6's words as well.
> `make_nature_methods_figures.py --only S1` builds it as `figureS1_measured_cost`.

S1, now fully measured from per-stage token accounting rather than estimated.

| stage | $/call | note |
|---|---|---|
| abstract screening | $0.0023 | **88% of cost is output** — the written rejection reason, not the abstract |
| full-text screening | $0.0138 | |
| coordinate parsing | $0.0059 | |
| annotation | $0.0211 | one call per study, not per study-column |

The practical claim: a full project runs for tens of dollars and hours, against months of person
time. For a methods journal this is not an aside — it is why the method matters.

### Discussion (~600 w)

Four beats, roughly 150 words each:

1. **The benchmark is a reference, not truth.** Manual meta-analyses have their own scope,
   resolution and internal consistency. Where this binds, it binds hard — dementia's aggregation
   explains its whole profile. **`vbm_of_substance_use` is the sharpest case and it is checked
   against the source paper** — see the box below; three of its six columns have no significant
   published result to recover, and a fourth does not reproduce.

> **⚠ `vbm_of_substance_use`: read the source paper before scoring it — 2026-09-09.**
>
> Hill-Bowen et al. 2022, *Drug Alcohol Depend* 240:109625 (PMID 36115222), 82 articles, **ALE**.
> Verified against the full text. The paper reports drug-specific meta-analyses for five classes
> and finds significance in only two:
>
> | column | paper's result | gold map (MKDA) | gold vs paper inputs |
> |---|---|---|---|
> | all drug classes | significant (mFG/vmPFC, ACC, insula) | 214 vox | — |
> | alcohol | significant (bilat. cingulate, L IFG, L postcentral) | 441 vox | 22 exp / 213 foci — **exact** |
> | nicotine | **significant** (multiple PCC) | **0 vox**, max z 0.935 | 18 exp / 114 foci — **exact** |
> | cannabis | *"failed to yield significant clusters"* | 0 vox | 12 vs 9 exp, 153 vs 37 foci |
> | opioids | *"failed to yield significant clusters"* | 0 vox | 13 vs 12 exp, 72 vs 70 foci |
> | stimulants | *"failed to yield significant clusters"* | 0 vox | 18 vs 24 exp, 109 vs 236 foci |
>
> **Three consequences, in order of how much they matter.**
>
> **(a) cannabis, opioids and stimulants have no reference result.** The published finding is a
> null. Scoring the pipeline against them measures agreement with nothing: dice is necessarily 0
> because the *expert* map has no suprathreshold voxel, and r² is a correlation between two
> sub-threshold density maps. **They belong outside the 35-column denominator** — a benchmark
> *scope* decision, not a metric one. This supersedes an earlier retraction of mine in the
> metric/map section: the corrected-map r² of 0.24–0.59 those columns gained is not recovered
> signal, it is two null maps agreeing about where the literature happens to put coordinates.
> That is a trap the unthresholded metric walks straight into, and worth one sentence of warning.
>
> **(b) nicotine is a reproduction failure, and a clean one.** The gold inputs match the paper
> exactly — 18 experiments, 114 foci — yet MKDA finds nothing where ALE found PCC clusters. Same
> data, different algorithm, and the reference result disappears. neurometabench re-analyses every
> project with MKDA; this is the one place we can *prove* that substitution changes whether the
> reference exists. It needs stating in the neurometabench data paper, not buried.
>
> **(c) cannabis and stimulants have coordinate-extraction discrepancies** against the paper's own
> counts (153 vs 37 foci; 109 vs 236). Independent of significance, and independent of the
> pipeline — this is the benchmark's own extraction. Worth an audit before either paper ships.
2. **Where the gain appears and where it does not.** Specific targets gain; diffuse ones do not. Two
   accounts (squishy target vs model limitation) remain unresolved; say so rather than choosing.
3. **What generalises.** The analysis-unit argument is not neuroimaging-specific; the *measurability*
   is. Any synthesis whose result unit lives below the paper has this problem and no instrument.
4. **Close on §9** — the forward-looking use case, once it lands.

---

## Extended Data / Supplementary

Everything below survives at full length outside the word count:

- **§3's cautionary case** — verbatim criteria that look faithful and fail. Strong material,
  sacrificed to the limit; Extended Data figure plus two sentences in the body.
- §2's controlled precision analysis (dementia's published rejection reasons).
- Per-project tables behind Figures 2–5.
- The four-tier registry and `verbatim → best` leakage measurement (**still unrun**).
- Full PRISMA-style funnel per project.

---

## Before submission

| item | state |
|---|---|
| §9 forward-looking close | **the gate.** scz_enigma is the live candidate; Elsevier blocker cleared |
| statistical reporting | done — cluster bootstrap + sign test wired into `compile_best_baselines.py` |
| cost and scale | done — measured, not estimated |
| `verbatim → best` | **measured 2026-09-09** — Supplementary S2. See the note below; the result is not comfortable |
| neurometabench citable | **needs a Zenodo DOI** before submission — reviewers will ask where the benchmark is |
| preprint | post simultaneously; NM desk-rejects fast, so the downside is bounded time only if the preprint is already out |

### Supplementary S4 — NeuroQuery: a term is not an analysis

`scripts/neuroquery_baseline.py` → `reports/neuroquery_baseline.csv` → `--only S4`. Added
2026-09-09 because every other baseline is a *search* baseline, which tests the pipeline against
the Neurosynth-style workflow it descends from but not against the current generation of automated
map generators. "Why not just ask NeuroQuery?" is the first question this paper's framing invites,
and it had no answer.

**Mean *R²* 0.075 against the pipeline's 0.574, behind on 32 of 32 columns, median gap +0.510.**

**Do not report only that.** A gap that large invites the fair suspicion that the comparison is
rigged, and the structure of *where* NeuroQuery fails is the actual finding:

| kind of column | examples | NeuroQuery *R²* |
|---|---|---|
| canonical cognitive term | executive function, working memory, problem solving, mental arithmetic | 0.24–0.31 |
| condition contrast | all three emotion-regulation contrasts | **0.002** |
| clinical group comparison | alcohol, dementia functional, PTSD grey matter | 0.002–0.022 |

Per project: executive function 0.216 and problem solving 0.184, everything else ≤ 0.043.

**NeuroQuery encodes term-level association.** Where a benchmark column essentially *is* a term it
does respectably; where the column is a contrast between conditions, or a between-group clinical
comparison, the model has no representation for the thing being asked and scores near zero. **That
is this paper's analysis-unit argument arriving from an independent direction**, and it is a better
use of the result than a win count.

> **But that explains NeuroQuery's internal pattern, not the level difference — checked
> 2026-09-09.** The search baseline is *also* driven by a term, so "term versus contrast" cannot by
> itself account for 0.075 against 0.476. Three mechanisms do, and all three were measured:
>
> **(a) The search baseline is not a term encoder. It retrieves the experts' own studies.** A
> targeted PubMed query returns real papers about that contrast, and their reported coordinates
> are pooled — so it inherits the actual empirical signal rather than predicting it. Measured
> across the 31 columns with a search arm, **the baseline retrieves a median 37% of the gold
> studies (mean 51%, range 11–95%), and 15 of 31 arms retrieve more than half.** It is a noisy
> superset of the expert studyset, not a text model. This is the biggest single reason and it
> should be stated wherever the baseline is described.
>
> **(b) r² rewards sharing the expert map's *form*, which every MKDA arm gets free.** Gold and all
> MKDA arms are ~94% exact zeros, non-negative, on the same mask, with typical non-zero magnitude
> ~2.2–2.7. NeuroQuery is 100% non-zero, signed (to −5.7) and about five times smaller in typical
> magnitude. Two sparse maps agreeing about which 94% of voxels are exactly zero correlate
> substantially before any anatomy matches. **Ranking each map and taking the same number of top
> voxels removes sparsity, sign and scale: NeuroQuery then reaches 43% of the best baseline rather
> than 16%, so r² overstates the gap 2.8×** (top-*k* dice 0.189 vs 0.436 vs 0.526 for the
> pipeline, n = 30). Panel b reports both, and the caption must not quote the r² gap alone.
>
> **(c) NeuroQuery is deliberately smooth and generic.** It predicts the *average* map associated
> with a term across all of neuroimaging — robust for exploration, blunt for one contrast — and it
> has one map per query, so it cannot express "drug cue > neutral in dependent users" at all.
>
> **What survives all three corrections:** on the form-fair measure NeuroQuery still trails
> (0.189 vs 0.436), and the term-versus-contrast pattern persists there too — executive
> function 0.610 and working memory 0.504 against ER `increase` 0.033 and dementia `functional`
> 0.000. So the finding is real; it is the *size* of the r² gap that is partly an artefact.

**Three things the caption must say.**

1. **Not like-for-like.** NeuroQuery answers a different and far cheaper question — no studyset, no
   screening, no extraction, milliseconds. This shows term-level prediction cannot substitute for
   contrast-level synthesis, *not* that NeuroQuery is poor at its own task. Say so; the alternative
   reads as a straw man.
2. **r² only.** NeuroQuery produces no FDR-corrected map, so dice would need a threshold with no
   error control behind it — forbidden by the metric/map rule. "NeuroQuery scores no dice" would be
   an artefact of the comparison.
3. **The queries were fixed before scoring.** They are written out in the script and were not
   revised against results. This matters more than usual here: S2 shows revising criteria against
   feedback is worth +0.031, and tuning these queries would be the same mistake in our own favour.
   Several column names are unusable as queries alone ("decrease", "functional", "alcohol"), so
   they were spelled out from the construct rather than derived automatically.

Also worth one line: corr(NeuroQuery *R²*, search-baseline *R²*) = 0.688, so NeuroQuery is
strongest on the same columns the search baseline finds easy. It is not capturing something
orthogonal that the pipeline misses.

### Supplementary S3 — how much of the low precision is a pool mismatch?

`--only S3`. Precision is the weakest headline number in the paper, and there are two candidate
explanations: the screener admits studies the experts rejected, or **the pool it screens is not
the pool the experts drew from**. Our PubMed query returns a different corpus than the one behind
the published review, so studies that could never have appeared in the expert list are counted as
false positives however well screening works.

Three projects have a fixed-pool arm — same criteria, same screening, over a pool assembled
without search-driven narrowing. Holding screening constant and swapping only the pool separates
the two explanations.

| stage | mean precision, search pool | mean precision, fixed pool | Δ |
|---|---|---|---|
| search | 0.106 | 0.276 | **+0.169** |
| abstract screening | 0.201 | 0.373 | **+0.172** |
| full-text screening | 0.349 | 0.491 | **+0.142** |

**At full-text screening the pool accounts for +0.142 of precision — 29% of what is achievable
there.** So a substantial share of the number that looks like a screening failure is a corpus
difference, and it is separable.

**Panel b carries the control that makes this claim, and without it panel a's gap would be
uninterpretable.** The fixed pool buys precision at essentially no cost to *recall* — mean Δ
+0.000 at abstract and **+0.007** at full-text. A change that raised precision by quietly
discarding borderline true positives would show up there as a recall loss, and it does not. This
is the panel a reviewer will look for.

One caveat to keep in the caption: the search-stage recall delta is −0.057, driven entirely by
emotion regulation at −0.261. At that stage the two arms are different corpora by construction, so
it is not a screening result; every later stage is ~0.

**Where this goes.** Supplement, with two sentences in Result 2 pointing at it — the precision
number cannot be quoted without it, since a reader will otherwise take 0.35 as the pipeline's
ceiling when 0.49 is reachable on a matched pool. n = 3 projects, which the caption must state.

### Supplementary S2 — mis-specification is the real risk, not overfitting

`scripts/compile_tier_progression.py` → `reports/tier_progression.csv` → `--only S2`. Mean *R²*
against the expert map per tier, paired within project.

**The end-to-end `verbatim → best` number is the wrong thing to quote, and an earlier draft of
this section quoted it.** Split into its two segments it says something different and more useful:

| segment | what actually changed | n | mean |
|---|---|---|---|
| verbatim → manual | the author, having seen reports, fixed **major oversights** — criteria that were simply mis-specified | 2 | **+0.221** |
| manual → best | criteria rewritten against the error reports in full — this is the overfitting | 2 | **+0.031** |

**Almost the entire rise is the cost of having got the criteria wrong, not the benefit of having
seen the answers.** The mean line over the three projects with all three tiers goes 0.378 → 0.525
→ 0.530: steep, then flat.

That converts the leakage objection into a **result**, and it is one of the more useful things the
paper can say to a methods audience: *the live danger in LLM-assisted screening is mis-prompting,
and it is roughly seven times larger than the danger of tuning.* Say plainly that this happened to
us — the verbatim criteria for dementia (0.112 → 0.358) and cue reactivity (0.413 → 0.624) were
not subtly suboptimal, they were missing something important, and nothing in the pipeline's output
announced it. That is the argument for the whole error-report loop existing.

**Both n = 2, so neither figure is more than an indication.** Only four projects were ever
hand-revised, and two of those (`dementia`, `vbm_of_ptsd`) register the same run at `manual` and
`best`, contributing no second-segment measurement. The overfitting estimate in particular rests
on cue reactivity (+0.014) and social (+0.048) alone. **Do not report it without the n.**

**Still worth computing before submission:** this is absolute *R²* by tier, not **margin over
baseline** by tier. Baselines do not move with tier so the margin should shift similarly, but that
should be computed rather than assumed — `compare_baselines_to_benchmark.py` already takes
`--tier`.

Caption caveats: emotion regulation and social have maps at fewer than three tiers (drawn as a
plain point, meaning missing data); `vbm_of_ptsd` registers one run at every tier and is drawn as
an open circle, meaning no progression is measurable rather than none occurred.

**Sequencing with the data paper.****Sequencing with the data paper.** neurometabench as a separate Data Resource paper is the right
split, but it must be at least preprinted with a DOI before or alongside this one. "In a paper we
have not submitted yet" is not an answer to "where is this benchmark".
