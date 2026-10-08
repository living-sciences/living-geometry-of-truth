# 002-theory-update — How does linear truth structure scale? (outline)

Follow-up study #2, run **after** 001-living-update. Analysis-only: reuses `001-living-update/results/` and
its `followup_summary.json` / activation artifacts. **~0 GPU-h** (CPU plotting/fitting). Run via:

```
python -m veritas.cli.main followup <run_dir> --instruction-file 002-theory-update.instruction.md --name theory-update
```

(reuses 001's `results/`). Leaves `report.md`, `followup_summary.json`, `result_card.json`
(schema `sai.followup.result_card/v1`), figures under `results/`.

## Question
From 001's artifacts, how do **probe accuracy**, **cross-dataset generalization**, and **best layer
(relative depth)** vary with **scale** (params), **family** (Llama/Qwen/OLMo), **generation** (2023→2025),
and **tuning** (base vs instruct)? Fit a simple relation and restate the paper's claim in updated words.

## Inputs (all from 001, no new extraction)
- Per-model per-layer probe accuracies (LR/MM/CCS), oracle + cross-dataset matrices.
- Per-model best-validation layer and the fixed 0.4-relative-depth cut.
- The C3 generalization number and mean off-diagonal accuracy per model.
- Optional MM flip-rates.
- Baselines read from disk: Llama-2-13B C3 = 0.9746 (MM), 7B<13B generalization gap (C19).

## Analyses
1. **Accuracy vs scale**: mean OOD generalization accuracy vs log10(params); separate markers by family;
   fit a curve (log-linear or saturating exponential `a - b·exp(-c·logP)`). Report slope + whether it
   saturates by ~8B. Compare to the paper's 7B→13B improvement (C19).
2. **Best layer vs depth**: best-validation probe layer as a **fraction of depth** per model; test whether
   truth is linearly represented at a roughly constant relative depth across families (the paper's Llama-2
   layers sit at ~0.35-0.41). Report mean ± sd relative depth and per-family values.
3. **Generalization by transformation**: per-model accuracy on negated sets (neg_cities, neg_sp_en_trans) vs
   affirmative — quantify whether the negation gap (C9/C12 geometry) shrinks in newer models; connect to the
   "Truth is Universal" 2-D-subspace hypothesis (2407.12831).
4. **Family/generation effects**: does 2025 (Qwen3, OLMo-3) beat 2024 (Llama-3.1, Qwen2.5) at equal scale?
   Is there a family-specific offset?
5. **Tuning axis**: Qwen2.5-7B vs Qwen2.5-7B-Instruct — does instruction tuning preserve / strengthen /
   distort the truth direction (accuracy delta, cosine similarity of MM directions, flip-rate delta)?
6. **Fitted relation + updated claim**: one sentence, e.g. "Truth is linearly represented at ~X% relative
   depth across Llama/Qwen/OLMo, with cross-dataset generalization Y and rising ~Z per 10× params, saturating
   near ~N B; negation geometry is [more/less] aligned than in Llama-2."

## Deliverables
- `report.md`: the fitted relation, per-axis findings, and the updated one-line claim, with every baseline
  cited to its 001 artifact.
- `results/` figures: (a) accuracy vs scale w/ fit; (b) best relative depth per model/family;
  (c) negation-gap-vs-generation.
- `result_card.json` (`sai.followup.result_card/v1`): headline = the fitted relation with its key number;
  ≤2 tables; ≤3 figures; metrics with 001-artifact provenance.

## Constraints / anti-goals
- Analysis only; **no new GPU jobs**, no re-extraction. If 001 lacks a needed number, say so — do not fabricate.
- All numbers trace to 001 artifacts or on-disk replication artifacts; disclose any gaps.
- Do not modify the run dir outside `followup/002-theory-update/`.
</content>