# REPLICATION DIRECTIVE (read this first)

Added by the replication orchestrator (not part of the original repo). Binding
guidance for reproducing this paper; the rest of the repo is ground truth for
datasets, probing logic, and evaluation.

## 1. Model weights — use LOCAL converted Llama-2 (no HF token)
The environment's HF token has NO access to gated meta-llama models. Do NOT try to
download `meta-llama/Llama-2-*` — it will 401. Instead, HF-format Llama-2 weights
have been converted locally and are available here:
- `Llama-2-7b-hf`  -> `/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache-local/Llama-2-7b-hf`
- `Llama-2-13b-hf` -> `/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache-local/Llama-2-13b-hf`
(13B is the paper's central model.) Point the repo at these via its `config.ini`
`weights_directory` / absolute local paths, or load with
`AutoModelForCausalLM.from_pretrained("<abs path>")` / nnsight
`LanguageModel("<abs path>")`. 70B is not available locally — treat the 70B rows
(scale-comparison) as an environmental limitation and say so; reproduce the 7B/13B
results in full.

## 2. Code gotchas in this repo (verified during prep)
- **`--device remote` is the DEFAULT** in the scripts (NDIF remote server). You MUST
  pass `--device cuda:0` (or set it in config) for every generate/probe script, or
  jobs hang on remote auth. Run everything locally on the assigned GPU.
- **nnsight is imported but missing from `requirements.txt`, and the code targets the
  OLD nnsight <=0.2.x API** (`with model.forward(...) as runner: runner.invoke(...)`,
  `runner.output[...]`), removed in nnsight >=0.3. When you build the venv, either pin
  an nnsight version compatible with this code (a 0.2.x release) OR port the handful
  of tracing call-sites to the current nnsight API. Verify a tiny forward pass +
  activation read works before running the full extraction.
- `experimental_outputs/*.json` ship EMPTY (`[]`) and `acts/` is empty — there are no
  precomputed activations or reference outputs in the repo. Grade against the PAPER's
  reported numbers (Table 2 probe accuracies; Figs 5, 9-11 for generalization and
  causal intervention), not any in-repo output file.

## 3. Scope priority (~1-2 GPU-h on 13B, forward-passes only)
1. Extract residual-stream activations for the true/false datasets (cities,
   sp_en_trans, larger_than, etc.) on Llama-2-13b; train mass-mean + logistic-
   regression probes; reproduce the linear-separability / probe-accuracy table.
2. Cross-dataset probe generalization (train on one, test on others).
3. Causal intervention: patch along the truth direction and show it flips the
   model's truth assessment (the repo's patching scripts, forced to cuda:0).
4. Scale comparison: add Llama-2-7b (also local) for the 7B-vs-13B trend; note 70B
   is unavailable (no local weights, no token).

Datasets all ship in the repo. Forward passes only — one GPU is plenty.
