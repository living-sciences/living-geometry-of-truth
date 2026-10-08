# 003-living-update — Do truth directions survive in the newest 2026 models?

Follow-up study for *The Geometry of Truth* (Marks & Tegmark, arXiv 2310.06824). This is the
**2026 model-refresh tick** that extends the prior living-update (001, which stopped at 2025 models)
to the newest 2026 flagships.

## Question

The paper established that truth is **linearly separable** in LLM residual streams: mass-mean (MM),
logistic-regression (LR) and CCS probes recover a truth direction that (a) separates true/false
statements (C1), (b) **generalizes across datasets** — train on `larger_than+smaller_than`, test on
`sp_en_trans` (headline claim **C3**) — and (c) is **causally load-bearing** (C2/C17). The paper studied
only Llama-2 (2023); the 001 living-update confirmed it held across 2024–2025 Llama/Qwen/OLMo models and
base→instruct. **Does it still hold on 2026 models — primarily the newest Qwen (Qwen3.5-9B-Base, Mar 2026),
and optionally Gemma-4-12B?** We hold the paper's method fixed, reuse every 2023–2025 result, and only
compute the new 2026 points, then re-aggregate all of them and extend the over-time figure to 2026.

## Approach

- **Reused the replicated codebase + the 001 living-update verbatim.** Probe classes (`probes.py` —
  `LRProbe`/`MMProbe`/`CCSProbe`), the medley/oracle/split protocol (split 0.8, seed 42, per-dataset
  centering) from `followup_analysis.py`, the seed-0 N≤1000 capped datasets, and `utils.get_pcs` for PCA
  are all reused. **All 2023–2025 per-model numbers are read from the 001 results on disk and merged, not
  recomputed** (`results/generalization_followup.json` contains the 5 prior new models + the 2 Llama-2
  anchors + the 2 new 2026 models = 9 models). The 5 prior new-model activation dirs are symlinked into
  `acts/`; only the 2026 models were freshly extracted.
- **Baselines are read from disk, never hand-typed.** The Llama-2-13B C3 anchor (MM **0.9746**, LR
  **0.9718**, CCS **0.9774**) comes from
  `run/replication/codebase/experimental_outputs/generalization_results.json` (model `llama-2-13b`, layer
  14, `oracle:False`); the C19/oracle common-grid recompute (0.769) from the 001 result.
- **The activation-extraction backend changed for the 2026 models (the one thing the instruction
  required).** GoT's original pipeline used nnsight `model.trace()`, and the 001 followup used nnsight
  with transformers 4.57. **nnsight cannot wrap the 2026 multimodal architectures** (`qwen3_5`,
  `gemma4_unified`). So for the 2026 models I loaded with plain transformers
  (`AutoModelForImageTextToText`, bf16, `device_map="auto"`) and read the residual stream via
  `output_hidden_states`. I verified that `hidden_states[L+1][:, -1, :]` (output of decoder layer `L`)
  is **identical** to a forward hook on the decoder `layers[L]` (max abs diff **0.0**), so the extracted
  point matches the nnsight `layers[L].output` convention used for the earlier models. Left padding makes
  `[:, -1, :]` the true last content token for every sequence.
- **A fresh venv was required.** The 001 venv (transformers 4.57.1) does not know `model_type` `qwen3_5`
  or `gemma4_unified`. I built `workspace/.venv` (Python 3.11.15) with **transformers 5.17.0**, torch
  2.14.0+cu130, numpy 2.4.6, pandas 3.0.5, matplotlib 3.11.1. All caches under
  `/net/projects2/...` (`HF_HOME`, `UV_CACHE_DIR`, `TMPDIR`); nothing written to `/home`.
- **Smoke-tested each 2026 architecture before the full run** (load + one forward + one activation read).
  Both loaded cleanly on the A100 80 GB; no fallback swaps were needed.
- **Extraction bounds (as specified):** last-token bf16 activations, the same 8 datasets
  (`cities, neg_cities, sp_en_trans, neg_sp_en_trans, larger_than, smaller_than, likely,
  cities_cities_conj`), N≤1000/dataset seed 0 (paired sets share indices; sp_en_trans/neg use all 354),
  ~6 sweep layers per model at relative depths {0.25, 0.375, 0.4, 0.5, 0.625, 0.75}. **Realized new
  activation size: 0.68 GB** (350 MB Qwen3.5 + 330 MB Gemma-4; cap 20 GB).
- **Probes** run at both the validation-selected best layer (max MM mean off-diagonal) and the fixed
  **0.4-relative-depth** layer (`round(0.4·n_layers)` — the paper's rule), over the common 8-dataset grid.
- **Causal flip-rate** = the same cheap single-MM-direction analog of C2/C17 the 001 study used, but
  implemented with **transformers forward hooks** (nnsight unavailable): the MM direction from the model's
  own `cities+neg_cities` acts, added to false / subtracted from true statements across the
  intervene→probe layers during a forward pass over the `sp_en_trans` few-shot prompt, with a 1×/2×/4×/8×
  magnitude sweep. Run for Qwen3.5-9B-Base.
- **Realized compute: well under 1 GPU-h on one A100 80 GB, forward passes only** (Qwen3.5 extraction ~41 s,
  Gemma-4 extraction ~35 s; probing ~2.5 min for both; flip-rate + magnitude sweep ~40 s; the rest is model
  load time).

## Models added this tick

| Model | Release | Params | Arch (`model_type`) | Layers | Hidden | 0.4-depth layer | Loaded? |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B-Base** (primary) | Mar 2026 | 9B | `qwen3_5` | 32 | 4096 | 13 | ✅ transformers |
| **Gemma-4-12B** (optional 2nd) | 2026 | 12B | `gemma4_unified` | 48 | 3840 | 19 | ✅ transformers |

Both are multimodal-unified architectures; both were loaded and hooked successfully via plain
transformers. The decoder is at `m.model.language_model.layers` (the instruction said `m.language_model`;
the actual path in transformers 5.17.0 differs — found by module introspection).

## Results

### Headline: the finding still holds on the newest Qwen, and is a partial exception on Gemma-4

**C3 — cross-dataset generalization** (`larger_than+smaller_than → sp_en_trans`). Anchor read from disk;
2026 values from this session.

| Claim / metric | Paper | Llama-2-13B (anchor, disk) | **Qwen3.5-9B-Base** | **Gemma-4-12B** |
|---|---|---|---|---|
| **C3 MM @ paper 0.4-depth layer** | 0.95–1.00 | **0.9746** (L14) | **1.0000** (L13) | **0.3927** (L19) |
| **C3 MM @ best-val layer** | 0.95–1.00 | 0.9746 | 0.9944 (L16) | 0.7514 (L36) |
| **C3 LR @ best-val layer** | 0.95–1.00 | 0.9718 | 1.0000 | 0.5565 |
| **C3 CCS @ best-val layer** | 0.95–1.00 | 0.9774 | 1.0000 | 0.5819 |
| **C19 mean off-diag (MM), best-val** | qual. | 0.769 (8-grid) | **0.8832** | 0.8362 (0.4822 @ L19) |
| Oracle same-dataset MM | 1.00 | 0.988 | 0.9775 | 0.9544 |
| Oracle same-dataset LR | 1.00 | 0.994 | 0.9881 | 0.9812 |
| Causal MM flip-rate (analog) | 7/8 (NIE 0.72/0.54) | — (`nie_table_13b.json`) | 0.017 @1×, **0.994 TRUE→FALSE @2×** | not run |

Anchor sources: C3/oracle from `run/replication/codebase/experimental_outputs/generalization_results.json`;
common-grid C19 from `run/followup/001-living-update/results/generalization_followup.json`; causal from
`run/replication/codebase/experimental_outputs/nie_table_13b.json`.

**Qwen3.5-9B-Base — the paper's finding holds, strongly.** At the paper's own 0.4-depth layer the MM probe
gets **C3 = 1.000** (LR 0.997, CCS 0.997); at its best-val layer LR and CCS hit 1.000 and MM 0.994. Its
**C19 mean off-diagonal (0.883) is the highest in the entire 2023–2026 ladder**, above even the strongest
2024/2025 model. The top-2-PC scatter of `cities` shows clean true/false separation (C1). The linear
truth direction is emphatically **not** a Llama-2 artifact — three years and four generations later, the
newest Qwen reproduces it better than the 2023 anchor.

**Gemma-4-12B — a genuine partial exception (reported honestly).** Truth is still **linearly separable
within each dataset** (same-dataset oracle MM 0.954, LR 0.981 — comparable to the anchor), so the raw
feature exists. But the property that *defines* the paper's headline — **cross-dataset generalization at
the canonical 0.4-depth layer — fails**: C3 MM is only **0.393** (LR 0.449, CCS 0.376, all near chance) at
L19. The MM off-diagonal only climbs at much deeper relative depths (0.808 at L30, 0.836 at L36), and even
at that best layer the specific C3 cell is 0.751 for MM and ~0.56 for LR/CCS. So on the Gemma-4 unified
multimodal architecture the truth direction is present locally but **does not organize into a single
dataset-transferable direction at the layer the paper's rule points to** — an architecture-dependent
limitation, not a data or extraction bug (oracle accuracy rules that out).

### Causal: the Qwen3.5 truth direction is load-bearing once magnitude is corrected

Like the whole Qwen family in 001, Qwen3.5-9B-Base's natural-1× MM flip-rate is ~0 (overall 0.017): its
TRUE/FALSE readout is near-saturated (mean p(TRUE)−p(FALSE) ≈ ±0.95), so a nudge scaled by the natural
`cities` separation barely moves it. Scaling the **same** direction recovers a strong effect:

| Qwen3.5-9B-Base MM intervention magnitude | 1× | 2× | 4× | 8× |
|---|---|---|---|---|
| overall flip-rate (on correct) | 0.017 | 0.494 | 0.497 | 0.497 |
| TRUE→FALSE flip-rate | 0.034 | **0.994** | 1.000 | 1.000 |
| mean p_diff, true→subtract | +0.219 | −0.411 | −0.669 | −0.616 |

At 2× the direction flips **99.4%** of TRUE statements to FALSE (mean p_diff crosses +0.95 → −0.41). The
FALSE→TRUE direction never flips at any magnitude (a known asymmetry — it is harder to steer a model into
asserting a falsehood is true). This is reported strictly as a cheap single-direction analog of C2/C17
(not the per-layer×per-token 12-cell NIE Table 2, which stays on the two Llama-2 anchors already on disk).

## Deviations & limitations

- **Decoder attribute path.** The instruction gave `dec = m.language_model`; in transformers 5.17.0 both
  2026 models expose the decoder at **`m.model.language_model.layers`**. Found by introspection; the
  `output_hidden_states` residual reading works exactly as instructed and matches a forward hook on
  `layers[L]` to 0.0.
- **Fresh venv (transformers 5.17.0), not the 001 venv.** transformers 4.57.1 does not know `qwen3_5` /
  `gemma4_unified`. All 2023–2025 numbers are reused from the 001 result on disk (not recomputed); only
  the 2026 models were computed this session.
- **nnsight not used for the 2026 models** (it cannot wrap `qwen3_5` / `gemma4_unified`), as the
  instruction anticipated. Extraction uses transformers `output_hidden_states`; the causal analog uses
  transformers forward hooks.
- **Gemma-4-12B is the optional second point.** Its weights were fully present (23 GB single-file
  safetensors) and it loaded/hooked cleanly, so it is included as a documented partial (not a silent skip).
  Its causal flip-rate was **not** run — optional, and its interpretation would be muddy given the weak
  cross-dataset generalization.
- **Causal flip-rate is a cheap single-MM-direction analog**, not the full NIE Table 2. The Qwen3.5 0-flip
  at 1× reflects readout saturation, not absence of causal structure (the magnitude sweep shows the
  direction is aligned).
- **Capped extraction (N≤1000, seed 0, 8 datasets, ~6 layers, bf16).** C19 is computed on the common
  8-dataset grid, matching the 001 protocol and the anchor's common-grid recompute (0.769). Extended paper
  datasets (`cities_cities_disj`, `companies/common_claim/counterfact`) were not re-extracted, per the
  storage cap.
- **Claims not extended, per the original methodology:** C4/C5 (70B logprobs), C6/C7/C16 (full
  activation-patching sweeps / Table 2), C11/C12 (layer emergence, secondary), C20 (few-shot).

## Artifacts

- `results/generalization_followup.json` — full per-model, per-layer generalization matrices (LR/MM/CCS
  medleys + oracle) for all 9 models (old reused-from-disk + the 2 new 2026 models); best-val and
  0.4-depth summaries.
- `results/flip_rates.json` — per-model MM flip-rate analog (001 models + Qwen3.5-9B-Base magnitude sweep).
- `results/truth_direction_over_time.png` — C3 vs release era extended to the 2026 points (both ringed;
  MM @ 0.4-depth shown as an x; anchor line at 0.9746; base↔instruct pair).
- `results/probe_acc_vs_scale.png` — mean OOD generalization (C19) vs log(params), 2026 models as stars.
- `results/pca_cities_grid.png` — top-2-PC `cities` scatter per model incl. both 2026 models (C1).
- `workspace/codebase/` — reused 001 scripts + this tick's new code: `smoke_qwen35.py`,
  `extract_qwen35.py`, `extract_mm2026.py` (generic 2026 extractor), `analyze_qwen35.py`,
  `flip_rate_qwen35.py`, `make_figures_003.py`; `pinned_versions_2026.txt`.
- `acts/qwen3.5-9b-base/`, `acts/gemma-4-12b/` — new-model activations (0.68 GB); the 5 prior new models
  are symlinked from 001.
