# Large files omitted from this repository

GitHub rejects files over 100 MB and discourages files over 50 MB, and the activation
caches this project produces run to many gigabytes. To keep the repository clonable, the
large model-derived artifacts below are not included. None of them are needed to read the
results; every one can be regenerated from the code and datasets that are in the repo.

Sizes are the on-disk sizes from the original run. Paths are relative to the repository root.

## Activation caches (residual-stream activations, `acts/`)

These directories hold per-statement, per-layer residual-stream activation tensors saved as
many small `*.pt` files (batches of 25 statements), produced by the authors'
`generate_acts.py`. They are the single largest artifact of the project and are excluded in
full (the directories and every `*.pt` inside them).

| Path | Size | What it is |
|------|------|------------|
| `run/replication/codebase/acts/` (Llama-2-7B and Llama-2-13B) | about 899 GB across 167,185 `*.pt` files (one dataset-layer-set such as `llama-2-13b/cities/` is about 9 GB; the `counterfact_true_false` set at ~32k statements across all layers dominates the total) | Full residual-stream activations for Llama-2-7B and Llama-2-13B across all layers of all 12 datasets, from the replication run. |
| `run/followup/001-living-update/acts/` | about 1.42 GB across 7,830 `*.pt` files | Last-token activations for the five 2024-2025 models (Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-7B-Instruct, Qwen3-8B-Base, OLMo-3-7B), capped at <=1000 statements per dataset across a ~6-layer sweep. |
| `run/followup/003-living-update/acts/` | about 0.60 GB across 3,240 `*.pt` files | Last-token activations for the 2026 models (Qwen3.5-9B-Base and Gemma-4-12B). |

In total the omitted activation caches are about **901 GB across roughly 178,000 `*.pt` files**.

**How to regenerate.** Point the authors' `run/replication/codebase/config.ini` at local
Llama-2 weight directories (the models are gated on Hugging Face), then run, for example:

```
cd run/replication/codebase
python generate_acts.py --model llama-2-13b --layers -1 \
  --datasets cities neg_cities sp_en_trans neg_sp_en_trans larger_than smaller_than \
  --device cuda:0
```

This writes `acts/<model>/<dataset>/layer_<L>_<idx>.pt`. The follow-up studies used the same
mechanism via each study's `workspace/` (see `extract_new.py` / `extract_qwen35.py` and the
study `instruction.md` for the exact model list, layer sweep, and per-dataset cap of 1000).
Extraction is forward-passes only over short single-sentence inputs and is cheap on one GPU
once weights are available.

## Dataset regeneration helper

| Path | Size | What it is |
|------|------|------------|
| `run/replication/codebase/datasets/counterfact.json` | 45,108,470 bytes (about 45 MB) | Raw CounterFact source used only to regenerate `counterfact_true_false.csv`, which is already present in the repo. Obtain it from the authors' repository (https://github.com/saprmarks/geometry-of-truth) under `datasets/` if you need to rebuild that CSV. The smaller helper `datasets/geonames.csv` (about 28 MB) is kept. |

## Other omitted items (not large-data, listed here for completeness)

These were left out as run machinery rather than for size, but are noted so the inventory is
complete:

- `run/replication/codebase.diff` (about 33 MB) was excluded. It is a git diff of the run
  directory in which more than 99% of the lines are entries for the omitted `acts/*.pt`
  tensors, so it is effectively an activation-cache manifest rather than a useful code diff;
  the actual code as run is present under `run/replication/codebase/`.
- The verbose engine transcripts (`*transcript*.jsonl`) and the grading signal file
  (`diligence_signals.json`) were excluded as internal run machinery.
- Python virtual environments (`.venv/`) and `__pycache__/` directories were excluded; see
  `.gitignore`.
