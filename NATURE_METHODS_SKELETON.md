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

```
best AVAILABLE   0.495 vs 0.394   Δ +0.101   95% CI [+0.045, +0.185]   sign test p = 2.2e-05
STRONGEST        0.495 vs 0.399   Δ +0.096   95% CI [+0.040, +0.179]   p = 1.2e-04
```

CI is cluster-bootstrapped over projects, not columns — a project's columns share a corpus, a
search and a screening run. Say so in one clause; it pre-empts the obvious reviewer objection and
costs nothing, since both intervals exclude zero.

### Metric consistency — unresolved

Figure 4 currently reports **r²** and Figure 5 **dice**, which is incidental rather than principled:
each was whichever column its source file led with. `baseline_vs_autonima.csv` carries dice,
pearson r *and* r²; `annotation_value.csv` carries dice and pearson r but **no r²**. So dice is the
only metric available to both without recomputation.

The headline is robust to the choice — same 30/35 columns win under all three:

| metric | autonima | baseline | Δ | 95% CI | sign test |
|---|---|---|---|---|---|
| r² | 0.495 | 0.394 | +0.101 | [+0.045, +0.185] | 2.2e-05 |
| **dice** | 0.423 | 0.315 | **+0.107** | **[+0.052, +0.164]** | 2.2e-05 |
| pearson r | 0.678 | 0.598 | +0.080 | [+0.035, +0.142] | 2.2e-05 |

**Recommendation: unify on dice**, and state in Methods that the result is unchanged under r² and
pearson r. Dice is available to both figures, its interval is the narrowest, and reporting the
robustness explicitly is stronger than picking one silently — "why this metric?" is otherwise a
free shot for a reviewer.

One argument against r² that turns out **not** to apply: r² discards sign, so an anti-correlated
map would score as well as a correlated one. Checked — 0 of 104 comparisons have negative pearson
r, so this is theoretical here. Do not use it as the justification.

Cost of switching: §7 and the poster figures both use r² throughout, so the text needs editing.
Not yet done.

### Result 5 — The gain comes from analysis selection, not paper selection — **Figure 5** (~350 w)

§6. **This is the paper.** Holding the study pool fixed, annotation still improves the map — so the
advantage is not explained by retrieving better papers.

**Median +0.037 dice, 26 of 35 columns improve.** Note this is a softer headline than Result 4's
+0.101 and measures a different thing; Result 5 must quote its own number rather than borrowing
Result 4's.

Single panel, deliberately. An earlier draft paired this with a slope plot of baseline vs annotated
dice across all 35 columns — 35 crossing lines in 89mm, unreadable, and carrying nothing the gain
distribution does not already show except absolute dice levels, which Figure 4 supplies.

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

### Result 6 — Cost and scale — **Figure 6** (~300 w)

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
   explains its whole profile.
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
| `verbatim → best` | designed, never run; needed for the leakage claim |
| neurometabench citable | **needs a Zenodo DOI** before submission — reviewers will ask where the benchmark is |
| preprint | post simultaneously; NM desk-rejects fast, so the downside is bounded time only if the preprint is already out |

**Sequencing with the data paper.** neurometabench as a separate Data Resource paper is the right
split, but it must be at least preprinted with a DOI before or alongside this one. "In a paper we
have not submitted yet" is not an answer to "where is this benchmark".
