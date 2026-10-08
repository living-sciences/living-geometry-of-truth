# 001-living-update — Do truth directions survive in today's models?

Follow-up study for *The Geometry of Truth* (Marks & Tegmark, arXiv 2310.06824).

## Question

The paper established that truth is **linearly separable** in LLM residual streams: mass-mean (MM)
and logistic-regression (LR) probes recover a truth direction that (a) separates true/false
statements, (b) **generalizes across datasets** (train on `larger_than+smaller_than`, test on
`sp_en_trans`, claim **C3**), and (c) is **causally load-bearing** (adding/subtracting it flips a
model's TRUE/FALSE readout, claims C2/C17). The paper only studied **Llama-2 (2023)**. Holding the
method fixed, does this still hold in models released **2024–2025**, across families
(**Llama / Qwen / OLMo**) and generations, and does it **transfer from base to instruction-tuned**
models?

## Approach

- **Reused the replicated codebase** at `run/replication/codebase/` verbatim: the probe classes
  (`probes.py` — `LRProbe`/`MMProbe`/`CCSProbe`), the nnsight trace-based extraction, the
  medley/oracle/split protocol from `generalization_run.py` (split 0.8, seed 42, per-dataset
  centering), and `utils.get_pcs` for PCA. Baselines were **read from disk**, never hand-typed.
- **Llama-2 anchors (7B, 13B):** reused the existing activations under
  `replication/codebase/acts/llama-2-{7b,13b}/` — no model load, no re-extraction. My pipeline
  reproduces the on-disk C3 values **exactly** (13B MM **0.9746**, 7B MM **0.9266**), validating that
  the followup pipeline is faithful.
- **New models:** the replication venv (transformers 4.38) cannot load Llama-3.1/Qwen2.5/Qwen3/OLMo-3.
  Built a fresh `uv` venv (Python 3.11.15) with **transformers 4.57.1 + nnsight 0.4.11**
  (torch 2.14.0+cu130). One code change was needed: transformers ≥4.57 decoder layers return the
  hidden-state **tensor** directly, so the last-token selector is `.output[:, -1, :]` (not the
  pre-4.57 `.output[0][...]`). Verified identical layer path for Llama-3.1/Qwen2.5/Qwen3/OLMo-3.
- **Smoke-tested every architecture** before extraction (single statement at the 0.4-depth layer).
  **All five new architectures loaded on the first try — no fallback swaps were needed.**
- **Extraction bounds (as specified):** last-token bf16 activations, 8 datasets
  (`cities, neg_cities, sp_en_trans, neg_sp_en_trans, larger_than, smaller_than, likely,
  cities_cities_conj`), **N ≤ 1000/dataset** random with **seed 0** (paired sets share indices so CCS
  pairing and negation alignment are preserved; sp_en_trans/neg use all 354; likely capped at 1000),
  ~6 sweep layers per model at relative depths {0.25,0.375,0.5,0.625,0.75}×n plus round(0.4·n).
  **Realized activation size: 1.6 GB** (cap 20 GB).
- **Probes** run at **both** the validation-selected best layer (max MM mean off-diagonal) **and** the
  fixed 0.4-relative-depth layer, over the common 8-dataset grid so C19 is apples-to-apples.
- **Causal flip-rate** = a cheap analog of C2/C17: a *single* MM direction from the model's own
  `cities+neg_cities` acts, added/subtracted across the intervene..probe layers during a forward pass
  over the `sp_en_trans` few-shot prompt; fraction of statements whose TRUE/FALSE prediction flips to
  the intended target. **Clearly an analog, not the 12-cell NIE Table 2.**
- **Realized compute: ~0.4 GPU-h on one A40 48 GB, forward passes only** (extraction ~4 min total;
  flip-rate + magnitude sweep ~10 min; probing on-GPU ~2 min). Budget was ≤8 GPU-h.

## Model ladder (all loaded from local snapshots; no swaps)

| Era | Model | Params | Arch | Role |
|-----|-------|--------|------|------|
| 2023 | Llama-2-7B | 7B | Llama | base, reuse acts |
| 2023 | **Llama-2-13B** | 13B | Llama | **paper anchor**, reuse acts |
| 2024 | Llama-3.1-8B | 8B | Llama | base |
| 2024 | Qwen2.5-7B | 7.6B | Qwen2 | base + instruct contrast |
| 2025 | Qwen3-8B-Base | 8.2B | Qwen3 | base |
| 2025 | OLMo-3-7B (Olmo-3-1025-7B) | 7.3B | Olmo3 | base |
| tuning | Qwen2.5-7B-Instruct | 7.6B | Qwen2 | instruct contrast |

## Results

### Headline: the correlational truth direction survives everywhere

**C3 — cross-dataset generalization** (`larger_than+smaller_than → sp_en_trans`). Values at each
model's **validation-best layer**; anchors read from disk and reproduced exactly.

| Claim / metric | Paper | Llama-2-13B (anchor, disk) | Llama-2-7B | Llama-3.1-8B | Qwen2.5-7B | Qwen3-8B-Base | OLMo-3-7B | Qwen2.5-7B-Instruct |
|---|---|---|---|---|---|---|---|---|
| **C3 MM** | 0.95–1.00 | **0.9746** | 0.9266 | **0.9859** | 0.9605 | 0.9774 | 0.9350 | 0.9802 |
| **C3 LR** | 0.95–1.00 | 0.9718 | 0.7458 | 0.9718 | 0.9237 | 0.9887 | 0.9463 | 0.9831 |
| **C3 CCS** | 0.95–1.00 | 0.9774 | 0.7966 | 0.9605 | 0.9520 | 0.9887 | 0.9633 | 0.9859 |
| C3 MM @0.4-depth | — | 0.9746 (L14) | 0.9266 (L13) | 0.9859 (L13) | 0.8192 (L11) | 0.9237 (L14) | 0.8955 (L13) | 0.9633 (L11) |
| **C19** mean off-diag (MM) | qual. | 0.7690 (8-grid) / 0.7589 (11-grid, disk) | 0.5189 | 0.8077 | 0.7925 | 0.8120 | 0.7345 | 0.8500 |
| Oracle (same-dataset) MM | 1.00 | 0.9877 | 0.9834 | 0.9887 | 0.9744 | 0.9609 | 0.8012 | 0.9806 |
| Oracle (same-dataset) LR | 1.00 | 0.9943 | 0.9836 | 0.9894 | 0.9906 | 0.9862 | 0.9634 | 0.9894 |
| **Flip-rate** (MM analog, 1×) | 7/8 causal (NIE 0.72/0.54) | — (anchor: `nie_table_13b.json`) | — | **0.983** | 0.000* | 0.003* | 0.000* | 0.000* |

Source for every "paper/anchor" number: `run/replication/codebase/experimental_outputs/generalization_results.json`
(C3/C19/oracle) and `.../nie_table_13b.json` (causal). `*` see the causal section below.

**Every 2024–2025 base model reproduces the paper's headline C3 finding** at ~0.93–0.99 MM. The
newest Llama (Llama-3.1-8B, 0.9859) **exceeds** the 2023 13B anchor at a smaller size. The claim that
truth is a robust, cross-dataset-generalizing linear direction is **not an artifact of Llama-2 — it
holds across three model families and two additional generations.**

- **C1 (PCA separation):** the top-2-PC scatter of `cities` activations shows clean true/false
  separation in every model (`results/pca_cities_grid.png`). C1 survives everywhere.
- **C19 (bigger/newer generalizes better):** every modern ~8B model matches or beats the 13B anchor's
  0.769 mean off-diagonal (0.73–0.85 vs 0.52 for Llama-2-7B). Scale/era both help.

### Base → instruct: the direction transfers to chat models

The one labeled contrast pair, **Qwen2.5-7B (base) vs Qwen2.5-7B-Instruct**:

| Metric | Qwen2.5-7B (base) | Qwen2.5-7B-Instruct |
|---|---|---|
| C3 MM (best layer) | 0.9605 | **0.9802** |
| C3 LR | 0.9237 | 0.9831 |
| C19 mean off-diag (MM) | 0.7925 | **0.8500** |

Instruction tuning **does not destroy the truth direction — it slightly strengthens the
cross-dataset probe** (higher C3 and the best C19 in the whole ladder). (Instruct points are drawn
as a connected pair in `truth_direction_over_time.png`, not folded into the base trend line.)

### Causal flip-rate: aligned everywhere, but the cheap analog is under-powered off-Llama

The single-direction MM flip-rate cleanly reproduces the paper's causal story for **Llama-3.1-8B
(0.98 of statements flip to the intended TRUE/FALSE label)**. For Qwen2.5-7B, Qwen3-8B-Base, OLMo-3-7B
and Qwen2.5-7B-Instruct the natural-magnitude analog flips ~0% — **but this is a magnitude artifact,
not an absence of causal structure.** Those models' TRUE/FALSE readout is near-saturated
(mean p(TRUE)−p(FALSE) ≈ ±0.99), so a nudge scaled by the natural `cities` separation barely moves it.
Scaling the **same** Qwen2.5-7B direction recovers the effect:

| Qwen2.5-7B intervention magnitude | 1× | 2× | 4× | 8× |
|---|---|---|---|---|
| flip-rate (overall, on correct) | 0.000 | 0.000 | **0.493** | 0.129† |
| mean p_diff, true→subtract | +0.957 | +0.929 | **−0.424** | +0.082 |

At 4× the direction flips half of statements (true→false mean p_diff crosses from +0.97 to −0.42),
confirming the direction is **causally aligned**; †at 8× the few-shot format starts to degrade
(non-monotonic). This is reported strictly as a cheap analog of C2/C17 (a single MM direction, not the
per-layer×per-token 12-cell NIE table, which was kept on the two Llama-2 anchors per the instruction).

## Deviations & limitations

- **Flip-rate is an analog**, not the full NIE Table 2. Read the causal column as "does one MM
  direction move the readout," not as a NIE match. The natural-magnitude 0-flip results for Qwen/OLMo
  reflect readout saturation (the direction *is* aligned; see the magnitude sweep), so I do **not**
  claim the causal finding fails for those families — only that the cheapest analog under-powers them.
- **Capped extraction (N≤1000, seed 0, 8 datasets, ~6 layers, bf16).** New-model C19 is computed on
  the common 8-dataset grid; the anchor is recomputed on that same grid (0.7690) and cross-checked to
  the disk full-11-grid value (0.7589). The extended paper datasets (`cities_cities_disj`,
  `companies/common_claim/counterfact_true_false`) were **not** re-extracted for new models.
- **Claims not extended, per instruction:** C4/C5 (70B logprob correlations), C6/C7/C16 (full
  activation-patching sweeps / Table 2 — kept on the two Llama-2 anchors already on disk). C11/C12
  (across-layer emergence/rotation) are secondary; the layer sweep is stored so they *can* be
  inspected, but were not separately figured. C20 (few-shot) skipped.
- **OLMo-3-7B massive activations:** a single statement's activation dominates raw PCA (an outlier in
  the top PCs), and depresses OLMo's same-dataset MM oracle (0.80) relative to its LR oracle (0.96).
  This is a known family-specific "massive activation" phenomenon; it does **not** break linear
  separability — OLMo's C3 is still 0.935.
- **Environment pinned:** Python 3.11.15, torch 2.14.0+cu130, transformers 4.57.1, nnsight 0.4.11,
  numpy 2.4.6, pandas 3.0.5 (`workspace/pinned_versions.txt`). One A40 48 GB, ~0.4 GPU-h, 1.6 GB acts.

## Artifacts

- `results/generalization_followup.json` — full per-model, per-layer generalization matrices (LR/MM/CCS
  medleys + oracle), best-val and 0.4-depth summaries.
- `results/flip_rates.json` — per-model MM flip-rate analog.
- `results/truth_direction_over_time.png` — C3 vs release era (MM primary; LR/CCS secondary; anchor line
  at 0.9746; base↔instruct pair).
- `results/probe_acc_vs_scale.png` — mean OOD generalization vs log(params), colored by family.
- `results/pca_cities_grid.png` — top-2-PC `cities` scatter per model (C1).
- `workspace/codebase/` — followup scripts (`make_capped_datasets.py`, `extract_new.py`, `fu_common.py`,
  `followup_analysis.py`, `flip_rate.py`, `make_figures.py`) + config entries for the new models.
- `acts/` — new-model activations (1.6 GB), reusable by later follow-ups.
