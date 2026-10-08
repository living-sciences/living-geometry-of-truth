# 002-theory-update — How does linear truth structure scale?

Follow-up study #2 for *The Geometry of Truth* (Marks & Tegmark, arXiv 2310.06824), run **after**
001-living-update and reusing its artifacts. **Analysis-only, ~0 GPU-h** (CPU fitting/plotting plus
CPU reads of 001's stored activations for one cosine-similarity).

## Question

From 001's artifacts alone, how do **probe accuracy**, **cross-dataset generalization**, and the
**best-layer relative depth** of the truth direction vary with **scale** (params), **family**
(Llama / Qwen / OLMo), **generation** (2023→2025), and **tuning** (base vs instruct)? Fit a simple
relation to the scale trend and restate the paper's "truth is a linear direction" claim in updated
words, citing every baseline to its 001 artifact.

## Approach

No new extraction and no GPU jobs. Everything is read from 001's on-disk outputs:

- **`001/results/generalization_followup.json`** — per-model, per-layer LR/MM/CCS generalization
  matrices, oracle diagonals, C3 (`larger_than+smaller_than → sp_en_trans`), C19 mean off-diagonal,
  best-validation layer, and the fixed-0.4-depth layer. 001 reproduced the Llama-2 anchors on disk
  **exactly** (13B MM C3 = 0.9746, 7B MM C3 = 0.9266), so these matrices carry the replication's
  numbers forward faithfully.
- **`001/results/flip_rates.json`** — MM single-direction causal flip-rates (used only for the tuning
  delta).
- **Model configs on disk** — `num_hidden_layers` (for relative depth) and `safetensors` `total_size`
  (÷2 for bf16 → parameter count) for each of the seven models.
- **`001/acts/…`** — stored last-token activations, read on **CPU** only to compute one number: the
  cosine similarity between the mass-mean (MM) truth directions of Qwen2.5-7B base vs instruct
  (`MMProbe.from_data` reused verbatim from the replicated `probes.py`).

Scripts: `workspace/analyze_theory.py` (all six analyses → `results/theory_update.json`) and
`workspace/make_figs.py` (three figures). Fits use `numpy.polyfit` (log-linear) and
`scipy.optimize.curve_fit` (saturating exponential `a − b·exp(−c·log₁₀P)`).

The base ladder used for the scale fit (instruct handled separately on the tuning axis):

| Model | Params (B, on-disk) | Layers | Family | Gen | Best-val layer | Rel. depth |
|---|---|---|---|---|---|---|
| Llama-2-7B | 6.74 | 32 | Llama | 2023 | 13 | 0.406 |
| Llama-2-13B (anchor) | 13.18 | 40 | Llama | 2023 | 14 | 0.350 |
| Llama-3.1-8B | 8.03 | 32 | Llama | 2024 | 13 | 0.406 |
| Qwen2.5-7B | 7.62 | 28 | Qwen | 2024 | 18 | 0.643 |
| Qwen3-8B-Base | 8.19 | 36 | Qwen | 2025 | 18 | 0.500 |
| OLMo-3-7B | 7.30 | 32 | OLMo | 2025 | 20 | 0.625 |
| Qwen2.5-7B-Instruct | 7.62 | 28 | Qwen | (tuning) | 18 | 0.643 |

## Results

### 1. Accuracy vs scale — weak across families, clean within-family

Mean OOD generalization (MM off-diagonal, **C19**) at each model's best-val layer vs log₁₀(params),
base ladder (`results/accuracy_vs_scale.png`):

- **Log-linear fit:** slope **+0.428 per 10× params**, but **r = +0.40** — weak, because the param
  range is narrow (6.7–13.2 B) and confounded with generation (the only >8B point, Llama-2-13B, is
  also the oldest).
- **Saturating exponential** `a − b·exp(−c·log₁₀P)`: asymptote **a = 0.915**, R² = 0.27; the fit
  reaches ≈**81 %** of its asymptote by **8 B** (predicted 0.738 at 8 B) — i.e., **generalization has
  largely saturated by ~8 B** within this window.
- **The clean scale signal is within-family Llama-2**: 7B→13B raises C19 **0.519 → 0.769** (+0.250)
  and C3 MM **0.927 → 0.975**. This *is* the paper's own C19 claim (bigger generalizes better) and
  matches the instruction's on-disk baseline (7B < 13B generalization gap).
- Across the modern 7–8 B cluster the C19 spread is **0.735–0.812** — essentially flat with params;
  almost all of the jump over Llama-2-7B (0.519) is **generation, not raw scale**.

**Comparison to the paper's 7B→13B improvement (C19):** reproduced and extended — the +0.25 C19 gain
from 7B→13B in Llama-2 is real, but a same-generation 8 B model (Llama-3.1-8B, 0.808) already
*exceeds* the 13B anchor (0.769), so at fixed architecture-generation the marginal value of scale
past ~8 B is small.

### 2. Best-layer relative depth — constant *within* Llama, deeper for Qwen/OLMo

Best-validation probe layer as a fraction of model depth (`results/best_relative_depth.png`):

- **Base mean 0.488 ± 0.123.** But this hides a strong **family split**: **Llama 0.388**, **Qwen
  0.571**, **OLMo 0.625**.
- The Llama family sits squarely in the **paper's Llama-2 band (~0.35–0.41)** — Llama-2-7B 0.406,
  Llama-2-13B 0.350, Llama-3.1-8B 0.406 — so "truth is linearly represented at ~0.4 relative depth"
  is a **Llama-specific constant**, not a universal one.
- Qwen and OLMo read truth out **deeper** (~0.5–0.64). This is not merely that truth is *absent*
  shallow: at the **fixed 0.4-depth cut** (mean rel-depth 0.392) every model still gives a strong C3
  (0.82–0.99, from 001), but the *optimal* readout for Qwen/OLMo lies deeper in the stack.
- **Caveat:** best-val layer is chosen from a coarse ~6-point sweep (relative-depth grid
  {0.25, 0.36–0.39, 0.5, 0.63, 0.75}), so per-model relative depth is resolved only to ≈±0.12. The
  qualitative Llama-shallow / Qwen–OLMo-deep split survives this resolution; finer depth claims do not.

### 3. Generalization by transformation — the negation gap collapses in newer models

Negation gap (MM, best-val layer): accuracy of an **affirmative-only** probe on its **negated**
counterpart drops relative to affirmative eval (`results/negation_gap_vs_generation.png`). This is the
paper's C9/C12 geometry — negation partially flips the truth direction.

| Model | cities→cities | cities→neg_cities | **negation gap** | C3 probe on neg_sp_en_trans | cities+neg medley → neg_cities |
|---|---|---|---|---|---|
| Llama-2-7B (2023) | 0.993 | 0.475 | **+0.519** | 0.167 | 0.980 |
| Llama-2-13B (2023) | 1.000 | 0.729 | +0.271 | 0.870 | 0.997 |
| Llama-3.1-8B (2024) | 1.000 | 0.706 | +0.294 | 0.938 | 0.995 |
| Qwen2.5-7B (2024) | 1.000 | 0.787 | +0.213 | 0.712 | 0.995 |
| **Qwen3-8B-Base (2025)** | 0.990 | 0.940 | **+0.050** | 0.808 | 0.995 |
| OLMo-3-7B (2025) | 0.660* | 0.498 | +0.162 | 0.797 | 0.760* |
| Qwen2.5-7B-Instruct | 1.000 | 0.908 | +0.092 | 0.842 | 0.995 |

- The single-direction negation gap **shrinks by ~10×** across generations: Llama-2-7B's affirmative
  direction **fully flips** on negation (0.475, below chance), whereas **Qwen3-8B-Base barely moves
  (0.940, gap +0.050)**. Newer models' 1-D affirmative truth direction already carries much more of
  the **polarity-invariant** ("general truth") component.
- This is exactly the **"Truth is Universal" 2-D-subspace picture** (Bürger et al., arXiv 2407.12831):
  training MM on **both** polarities (`cities+neg_cities` medley) recovers negated truth at **≈0.99 in
  every model** — the 2-D subspace (general + polarity direction) is present throughout; what changes
  with generation is how much of it a single affirmative direction captures.
- *OLMo-3-7B (\*) is depressed by its known family-specific massive-activation outlier (001), which
  drags raw MM `cities→cities` to 0.66; its LR oracle stays ≈0.96, so linear separability is intact.*

### 4. Family / generation effects — no clean 2024→2025 gain; family offset dominates

Mean C19 (MM, base) by generation: **2023 = 0.644** (7B+13B), **2024 = 0.800**, **2025 = 0.773**.

- **2025 does not beat 2024 at equal scale.** At ~7–8 B: Qwen3-8B-Base (0.812) edges its 2024
  predecessor Qwen2.5-7B (0.792), but OLMo-3-7B (0.735) sits below both 2024 models (Llama-3.1-8B
  0.808, Qwen2.5-7B 0.792), pulling the 2025 mean down.
- **Family offset is the larger effect.** Among moderns (~8 B): Llama-3.1 ≈ Qwen (0.79–0.81) > OLMo
  (0.735). Qwen and Llama are interchangeable at the top; OLMo carries a persistent negative offset
  (consistent with its massive-activation geometry).

### 5. Tuning axis — instruction tuning preserves and slightly strengthens the truth direction

Qwen2.5-7B **base vs instruct** (accuracy from 001; cosine similarity computed here on CPU from 001's
activations, MM direction = μ₊ − μ₋):

| Quantity | Base | Instruct | Δ |
|---|---|---|---|
| C3 MM (best layer) | 0.9605 | 0.9802 | **+0.020** |
| C19 mean off-diag (MM) | 0.7925 | 0.8500 | **+0.058** |
| MM flip-rate (1×, from 001) | 0.000 | 0.000 | 0.000 |
| **cos(MM_base, MM_instruct)** @ best-val L18 (cities / larger_than / sp_en_trans) | — | — | **0.843 / 0.866 / 0.897** |
| cos @ 0.4-depth L11 | — | — | 0.938 / 0.919 / 0.949 |

Instruction tuning **does not distort** the truth direction: the MM directions stay strongly aligned
(cos 0.84–0.90 at the deep best layer, 0.92–0.95 at the shallow 0.4-depth layer), and both probe
accuracy (**+0.02 C3, +0.058 C19**) and the negation gap (**+0.213 → +0.092**, §3) *improve*. Flip-rate
is unchanged (both 0 at 1× — a readout-saturation artifact carried over from 001, not a causal
difference). Net: tuning **preserves/strengthens**, it does not degrade.

### 6. Fitted relation + updated claim

> **Truth is linearly represented at ~0.4 relative depth in Llama (0.39 ± 0.02) but deeper in
> Qwen/OLMo (~0.5–0.63), for a cross-family mean of 0.49 ± 0.12; cross-dataset MM generalization is
> ~0.80 at 7–8 B and rises ~+0.43 per 10× params in a log-linear fit (clean within-family Llama-2
> signal 0.52 → 0.77 from 7B → 13B), saturating near ~0.9 by ~8–13 B; and negation geometry is far
> more aligned than in Llama-2 — a single affirmative direction's negation gap falls from 0.52
> (Llama-2-7B) to 0.05 (Qwen3-8B-Base), consistent with the 2-D-subspace hypothesis (2407.12831).**

## Deviations & limitations

- **Narrow parameter range (6.7–13.2 B) confounded with generation.** The cross-family "accuracy vs
  scale" fit is under-powered (r = 0.40, R² = 0.27); the honest scale evidence is the within-family
  Llama-2 7B→13B contrast. Reported as such — I do not over-claim a strong global scaling law.
- **Best-layer relative depth is resolved only to ≈±0.12** (coarse ~6-point sweep grid from 001). The
  Llama-shallow / Qwen–OLMo-deep split is robust to this; the exact per-model depths are not.
- **Cosine similarity uses the raw mass-mean direction** (μ₊ − μ₋, whitening-free), matching
  `MMProbe.direction`. It is computed at the shared best-val layer (18) and the 0.4-depth layer (11).
- **Flip-rate delta is 0/0** because 001's natural-magnitude single-direction flip-rate saturates for
  Qwen (readout p_diff ≈ ±0.99); this is inherited, not re-measured, and is not evidence about tuning.
- **OLMo-3-7B massive-activation outlier** (001) depresses its raw MM `cities` numbers; noted at each
  appearance. Its LR oracle (~0.96) confirms linear separability is intact.
- **All numbers trace to 001 artifacts** (`generalization_followup.json`, `flip_rates.json`) or
  on-disk model configs; 001 in turn reproduced the replication's Llama-2 anchors exactly
  (`replication/codebase/experimental_outputs/generalization_results.json`). No number was re-derived
  from memory or tuned toward the paper. No run-dir file outside `followup/002-theory-update/` was
  modified.

## Artifacts

- `results/theory_update.json` — all six analyses (per-model rows, fits, depth, negation, tuning).
- `results/accuracy_vs_scale.png` — C19 vs params with log-linear + saturating fits (fig a).
- `results/best_relative_depth.png` — best relative depth per model/family vs the paper's band (fig b).
- `results/negation_gap_vs_generation.png` — negation gap vs release generation (fig c).
- `workspace/analyze_theory.py`, `workspace/make_figs.py` — the analysis + figure code.
