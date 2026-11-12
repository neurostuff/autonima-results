---
name: screen-and-select
description: In one pass, screen studies at full text against numbered criteria and, for each included study, decide for every analysis and every meta-analytic target whether it belongs. Writes one JSON line per study. Use when given a cbma-review full-text batch whose skill field is "screen-and-select" (combined mode).
---

# Screen and select, in one pass

This batch combines two judgments that are otherwise separate stages:
- **full-text screening** of the study: the rules of `../screen-studies/SKILL.md`, full-text
  stage;
- **analysis selection** for each target: the rules of `../select-analyses/SKILL.md`.

Read both skill files first. Their rules apply unchanged; this file only says how the
batch and your output combine them.

## The batch

- `criteria` and `instructions`: the full-text screening criteria and guidance, `I1`,
  `E1`, ...
- `selection_criteria` and `selection_instructions`: the global selection criteria
  `GI1`, `GE1`, ..., and guidance for every target.
- `targets`: `{name: {description, criteria, instructions}}`, as in selection.
- `items`: one per study. Each has `text_file` (read it), and `analyses`: every analysis
  the study reports, with `analysis_id`, name, description and point count.

## For each study

1. **Screen it** exactly as at the full-text stage: give every screening criterion a
   state, decide `include`, `exclude` or `incomplete`, and quote your evidence.
2. **If it is `include`,** judge every analysis against every target, exactly as in
   selection: every global and target criterion gets a state, and `include` is true or
   false.
   - A study whose analyses are all ineligible for every target is still screened on
     its own criteria. The ledger then records it as excluded for having no eligible
     analysis.
   - Do not change the screening decision to anticipate this.
3. **If it is `exclude` or `incomplete`,** give no analysis decisions.

## Output

One JSON line per study, written to the batch's `output` path:

```json
{"pmid": "12345678", "decision": "include", "criteria": {"I1": "met", "E1": "not_met"}, "reason": "...",
 "evidence": [{"criterion": "I1", "quote": "exact text from text_file"}],
 "analyses": [{"analysis_id": "12345678-ana_1", "target": "t1", "include": true,
               "criteria": {"GI1": "met", "GE1": "not_met", "I1": "met"}, "reason": "..."}]}
```

- **An included study lists every analysis × target pair.** The ledger accepts a study
  only when its screening and all of its pairs are complete and consistent. Otherwise
  the whole study stays pending.
- **Reasons:** at most 200 words for the study and 80 per analysis decision.
- **An item you cannot judge:** leave its line out; it is retried.

Write the file once, when all items are done. Reply with only the number of lines
written.
