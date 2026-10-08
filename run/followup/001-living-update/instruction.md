# 001-living-update — Do truth directions survive in today's models?

Follow-up study for *The Geometry of Truth* (Marks & Tegmark, arXiv 2310.06824). Run via:

```
python -m veritas.cli.main followup \
  /net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run \
  --instruction-file <this file> --name living-update
```

This produces `followup/001-living-update/` inside the run dir. **Reuse** the replicated code at
`.../run/replication/codebase/` and the existing Llama-2 activations. Leave `report.md`,
`followup_summary.json`, `result_card.json` (schema `sai.followup.result_card/v1`), and figures under
`results/`.

---

## Question

Do the paper's core findings — **linearly separable truth directions** (mass-mean & LR probe accuracy,
**cross-dataset generalization**, and **causal intervention flip-rate**) — still hold in models released
2024-2025, across families (Llama/Qwen/OLMo) and generations, and do they transfer from base to
instruction-tuned models? The paper studied Llama-2 (2023) only. We hold the method fixed and vary the
model along an era/family/scale/tuning ladder.

---

## Claims this study UPDATES (IDs, paper value, replicated value, on-disk artifact)

Baselines are read **from disk** (originals), never hand-typed. All under
`.../run/replication/codebase/experimental_outputs/` unless noted.

| Claim | Paper value | Replicated (Llama-2-13B) value + artifact |
|-------|-------------|-------------------------------------------|
| **C3** (headline generalization): LR/MM/CCS trained on larger_than+smaller_than → sp_en_trans | 0.95–1.00 | LR **0.9718**, MM **0.9746**, CCS **0.9774** — `generalization_results.json` (oracle:false, model llama-2-13b, layer 14) |
| **C1** (PCA separation, top-2 PCs) | qualitative separation | match — `figures/*pca*` (13B, layer 14) |
| **C8** (cities PC1 separates other datasets) | qualitative | match — `figures/cross_basis.png` |
| **C9** (cities vs neg_cities axes ~orthogonal, mid layer) | qualitative | match |
| **C10** (larger vs smaller antipodal at 13B) | qualitative | partial (13B antipodal; 70B n/a) |
| **C13** (train on statement+opposite improves OOD) | qualitative | match — `generalization_results.json` |
| **C14** (MM<LR/CCS at 7B, ~= at 13B) | qualitative | match (7B MM avg **0.566**) |
| **C15** (likely-trained probes: good only where prob~truth) | qualitative | partial |
| **C18** (13B generalization matrix) | figure | partial — `figures/llama-2-13b_generalization.png` |
| **C19** (probes generalize better for larger models) | qualitative | match (13B mean off-diag **0.76** > 7B) |
| **C2** (MM more causal than LR/CCS, 7/8) — *anchor only* | 7/8 | match — `nie_table_13b.json` |
| **C17** (likely/MM intervention NIE) — *anchor only* | 0.70 / 0.54 | match |

Oracle (same-dataset) probe accuracy baselines: 13B cities/sp_en_trans/larger_than = **1.00** (LR & MM);
7B cities LR **0.99**, MM **0.9933**. Read from `generalization_results.json` (oracle:true entries).

## Claims this study does NOT update, and why

- **C4, C5** (70B logprob correlations): 70B-only, and forward-only budget. Not extended. (Optionally, a
  7B/13B logprob analog can be computed with `logprobs.py`, but report it as an analog, not a paper-value match.)
- **C6, C7, C16** (activation-patching maps / full Table 2): expensive per-layer×per-token causal sweeps.
  Keep on the two Llama-2 anchors (already done). Do **not** run the full patching sweep on new models.
- **C11, C12** (across-layer emergence / rotation): qualitative and layer-path-sensitive. Extract the layer
  sweep so these *can* be inspected on new models, but they are secondary; the headline metrics are C3/C1/C19.
- **C20** (few-shot baseline): optional; not a truth-direction claim. Skip unless time remains.

---

## Model ladder — verified 2026-09-09 against local cache (no HF token, no network probe)

Verification = read `config.json` from the on-disk snapshot; weights = multi-shard safetensors present.
Gated = HF status only; **local presence makes gating irrelevant for loading**.

| Era | HF id | Params | Arch class | Layers | Hidden | Gated | Local path |
|-----|-------|--------|-----------|--------|--------|-------|-----------|
| 2023 | meta-llama/Llama-2-7b-hf | 7B | LlamaForCausalLM | 32 | 4096 | yes (have) | `.../hf-cache-local/Llama-2-7b-hf` — **reuse acts** |
| 2023 | meta-llama/Llama-2-13b-hf | 13B | LlamaForCausalLM | 40 | 5120 | yes (have) | `.../hf-cache-local/Llama-2-13b-hf` — **reuse acts (ANCHOR)** |
| 2024 | meta-llama/Llama-3.1-8B | 8B | LlamaForCausalLM | 32 | 4096 | yes (local) | `shared_models/hub/models--meta-llama--Llama-3.1-8B` |
| 2024 | Qwen/Qwen2.5-7B | 7B | Qwen2ForCausalLM | 28 | 3584 | no | `shared_models/hub/models--Qwen--Qwen2.5-7B` |
| 2025 | Qwen/Qwen3-8B-Base | 8B | Qwen3ForCausalLM | 36 | 4096 | no | `shared_models/hub/models--Qwen--Qwen3-8B-Base` |
| 2025 | allenai/Olmo-3-1025-7B | 7B | Olmo3ForCausalLM | 32 | 4096 | no | `shared_models/hub/models--allenai--Olmo-3-1025-7B` |
| tuning | Qwen/Qwen2.5-7B-Instruct | 7B | Qwen2ForCausalLM | 28 | 3584 | no | `shared_models/Qwen/Qwen2.5-7B-Instruct` |
| scale (opt.) | Qwen/Qwen2.5-32B-Instruct | 32B | Qwen2ForCausalLM | 64 | 5120 | no | `HF_HOME/hub/models--Qwen--Qwen2.5-32B-Instruct` — **A100 80GB only** |

`shared_models` = `/net/projects2/chai-lab/shared_models`; `HF_HOME` =
`/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache`. Resolve each hub id to its snapshot at
`.../hub/models--<org>--<name>/snapshots/<hash>/` and pass that absolute dir as `weights_directory`.

**Substitutions / drops (see research_notes.md §4):** gpt-oss-20b — local dir empty → **dropped**.
OLMo-2 not on disk → **OLMo-3 substitutes**. Gemma-3-12B not local; `gemma-3-27b-it` is a multimodal
wrapper (nested `language_model.layers`) → **use `google/gemma-2-9b-it` (Gemma2ForCausalLM, 42 layers,
ungated) if a Gemma point is wanted, else skip**. Mistral local as instruct only. **Ungated fallbacks for
any arch that won't load:** Qwen2.5-7B(-Instruct), Meta-Llama-3-8B (base), gemma-2-9b-it.

**Base vs instruct decision:** the paper uses base models → the 6-model trend is **base only**. Add the one
labeled contrast pair **Qwen2.5-7B vs Qwen2.5-7B-Instruct** to test whether the truth direction transfers to
chat models. Do not fold instruct points into the base trend line.

---

## Environment (two venvs — this is the key constraint)

The replication venv (`.../run/.venv`: Python 3.11.15, torch 2.2.2+cu121, **transformers 4.38.2**,
**nnsight 0.2.21**) **cannot load Llama-3.1 / Qwen3 / Olmo3 / Gemma** (Llama-3.1 config has
`rope_scaling.rope_type="llama3"`, unknown to transformers 4.38). Therefore:

1. **Llama-2 anchors**: reuse existing activations at `.../codebase/acts/llama-2-{7b,13b}/` — no
   re-extraction, no model load. All layers already stored; use layer **14** (13B) and **13** (7B).
2. **New models**: create a fresh venv under the followup dir with `uv venv` (Python ≥3.11) and install a
   **recent transformers (≥4.57 to cover Olmo3) + a compatible recent nnsight (≥0.4)**; pin exact versions
   once all six architectures load. `HF_HOME` must point at the cache above; **never write to /home**.
3. **Smoke-test every architecture before full extraction** (see step 1 below). Any arch that fails → swap
   in an ungated fallback and record the swap in the report.

**Extraction code is drop-in.** `generate_acts.get_acts` already uses the stable trace API:
```python
with model.trace(statements, remote=False):
    for layer in layers:
        acts[layer] = model.model.layers[layer].output[0][:, -1, :].save()
```
Per-arch layer path: **Llama-2/3.1, Qwen2.5, Qwen3, Olmo-3 all use `model.model.layers[L].output[0]`**
(identical) — no code change beyond `config.ini` entries. (Gemma-2, if used: same path; Gemma-3: nested
`model.model.language_model.layers` — avoid.) Load with `LanguageModel(weights_directory,
torch_dtype=t.bfloat16, device_map="auto")` and always pass `--device cuda:0` (scripts default to remote NDIF).

---

## Storage & size caps (pre-specified, hard)

- **20 GB hard cap** on new activation outputs (expected ~2-3 GB). Write under
  `followup/001-living-update/acts/` (set `--output_dir` there or symlink from the codebase).
- **N ≤ 1000 statements/dataset**, random with fixed **seed 0**; sp_en_trans / neg_sp_en_trans (355) use all;
  **likely capped at 1000** (native 28 920 — never extract in full).
- **Store only the layer sweep**, not all layers: relative depths {0.25, 0.375, 0.5, 0.625, 0.75}×n_layers
  (rounded) **plus** the paper-rule layer `round(0.4·n_layers)`. ~6 layers/model.
- **Last-token, bf16** only. Budget check: 1000×4096×2 B ≈ 8 MB/dataset-layer → 8 datasets × 6 layers ×
  6 models ≈ **2.3 GB**. State the realized GB in the report.
- **Datasets (8, REUSE `.../codebase/datasets/`)**: cities, neg_cities, sp_en_trans, neg_sp_en_trans,
  larger_than, smaller_than, likely, cities_cities_conj. Never "find X elsewhere".

---

## Per-model computation (the ladder)

For each new model (Llama-2 anchors: read existing acts instead of extracting):

1. **Smoke test** (once per arch, before extraction): load model on cuda:0; run
   `get_acts(["The city of Paris is in France."], model, [round(0.4*n_layers)], remote=False)` and print the
   activation shape (`[1, hidden]`). If it errors, swap to a fallback and note it.
2. **Extract** last-token activations for the 8 datasets at the ~6 sweep layers, N≤1000/dataset, bf16, into
   `followup/001-living-update/acts/<model>/<dataset>/layer_<L>_<idx>.pt` (same layout `utils.collect_acts`
   expects). Add a `config.ini` section per model (name, `weights_directory`=snapshot abs path,
   `probe_layer`=round(0.4·n), `intervene_layer` optional, `noperiod=False`).
3. **Probes** (`probes.py`, reuse verbatim): train LR + MM on each train medley
   {cities, cities+neg_cities, larger_than, larger_than+smaller_than, likely} and CCS on the two paired
   medleys; evaluate on all 8 test datasets; compute oracle (same-dataset) accuracies. Do this **at the
   validation-selected best layer** (highest mean OOD accuracy on held-out medleys) **and** at the fixed
   0.4-relative-depth layer. Record both. This yields C3, C13, C14, C15, C18, C19 per model.
4. **Cross-dataset generalization matrix** (8×train×technique) per model → the C18 analog + C3 cell
   (larger_than+smaller_than → sp_en_trans) + mean off-diagonal for C19.
5. **PCA visuals** (`utils.get_pcs` / `dataexplorer` logic): top-2-PC scatter for cities (C1), cities basis
   projected onto other sets (C8), cities+neg_cities at mid layer (C9), larger+smaller (C10). One combined
   PCA panel per model is enough; these support the qualitative claims.
6. **Causal flip-rate (optional, only if per-model budget allows)**: on sp_en_trans, extract the mass-mean
   direction from cities+neg_cities, add/subtract it at the intervene layer during a forward pass, and record
   the fraction of statements whose predicted TRUE/FALSE label flips (a cheap analog of C2/C17 — a single
   direction, not the full 12-cell Table 2). Report as "MM flip-rate", clearly an analog. If budget is tight,
   run this for Llama-3.1-8B and Qwen2.5-7B only and say so.

---

## The over-time figure

Produce **`results/truth_direction_over_time.png`**: x-axis = model release date (or ordinal era
2023→2025), y-axis = **C3 generalization accuracy** (larger_than+smaller_than → sp_en_trans, MM probe) with
LR/CCS as secondary series; one marker per model, Llama-2-13B labeled as the paper anchor with a horizontal
line at the replicated **0.9746** (MM). Overlay the base↔instruct pair as a connected pair. Secondary figure
`results/probe_acc_vs_scale.png`: mean OOD generalization accuracy vs log(params), colored by family.

---

## Environment constraints (hard)

- **1 GPU/study**, saturated SLURM. A40 48GB for ≤13B (bf16 fits). Qwen2.5-32B needs an **A100 80GB**
  (~64 GB load) — take it only if a card is free; justify in the report. **Forward passes only.**
- All outputs and caches under `/net/projects2/chai-lab-models/haokunliu/...`; `HF_HOME` as above.
  **Never write to /home** (50 GB quota).
- **≤ ~8 GPU-h** for this study. Expected ~3-4 GPU-h (extraction ~1.5-2.5h; optional flip-rate ~1h).

---

## Deliverables (leave in `followup/001-living-update/`)

- **`report.md`** with a comparison table: one row per updated claim, columns
  **[claim, original paper value, replicated Llama-2-13B value + artifact file, then one column per new
  model]**. State the realized activation GB, the venv versions actually pinned, which models loaded vs were
  swapped, and which layer (best-val and 0.4-depth) each number came from.
- **`followup_summary.json`** — machine summary (models run, layers, per-model C3/C19 numbers, flip-rates).
- **`result_card.json`** (schema `sai.followup.result_card/v1`): headline with the key number (e.g.
  "MM truth-probe cross-dataset generalization (larger+smaller→sp_en_trans) = X on the newest model vs 0.9746
  Llama-2-13B"); status; 1-5 metrics each with baseline + provenance (the on-disk artifact path);
  ≤2 tables; 0-3 figures under `results/`; notes.
- **Figures** under `results/` (the two above + optional per-model PCA panel).

---

## ANTI-GOALS (read before running)

- **NEVER background a long job and end your turn to "wait" for it.** Run extraction/probing inline and
  observe it finish. No fire-and-forget.
- **No token budgets, no "I'll stop here to save context" behavior.** Do the ladder.
- **Every new number must come from THIS session's execution.** Do not invent or extrapolate model numbers.
- **All original/replicated baselines are READ FROM DISK** (paths above) — never hand-type a paper value
  from memory; cite the artifact file for each.
- **Disclose holes and any hand-coded values explicitly** — if a model is swapped, a claim skipped, or the
  flip-rate not run for some models, say so plainly in `report.md`.
- **Do not modify the replication run dir** outside `followup/001-living-update/`. Reuse the codebase and the
  Llama-2 activations read-only.
- **Bound extraction** as specified (N≤1000, ~6 layers, bf16, 20 GB cap) and **state the realized cost**.
</content>