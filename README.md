# The Geometry of Truth, replicated and kept alive

This repository is a *living paper*: a faithful replication of a published result, plus a
growing set of small extension studies that re-ask the paper's questions against newer models
and keep its findings current. The goal is simple. A finding should not be frozen at the moment
of publication. It should be checked again as the field moves, and the checks should be open for
anyone to read and rerun.

## The paper

**The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of
True/False Datasets.** Samuel Marks and Max Tegmark. Published as a conference paper at COLM 2024.

- arXiv: [2310.06824](https://arxiv.org/abs/2310.06824)
- arXiv DOI: [10.48550/arXiv.2310.06824](https://doi.org/10.48550/arXiv.2310.06824)
- Original code: https://github.com/saprmarks/geometry-of-truth

The paper argues that, at sufficient scale, large language models represent the truth of a simple
factual statement as a *linear* direction in their residual stream. It shows this three ways: true
and false statements separate along the top principal components of a hidden state; linear probes
trained on one dataset transfer to very different datasets; and adding or subtracting a probe
direction inside a small set of hidden states flips the model's TRUE/FALSE judgement. It introduces
mass-mean (difference-in-means) probing and shows that it matches logistic regression on accuracy
while giving directions that are more causally implicated in the model's output.

## Living page

A readable, continuously updated summary of this work lives at
https://livingscience.ai/safety/living-geometry-of-truth

## What this repo contains

Everything sits under `run/`.

### `run/replication/`

A from-scratch reproduction of the paper's core results on locally converted Llama-2 weights
(7B and 13B; the 70B rows are noted as an environment limitation because no local 70B weights were
available). Holding the authors' method fixed, it reproduces the linear PCA separation of true and
false statements, the cross-dataset probe generalization (training on `larger_than + smaller_than`
reaches mass-mean accuracy 0.9746 on `sp_en_trans` at 13B), the gain from training on statements
together with their negations, the `likely` plausibility control, the activation-patching
localization, and the headline causal result that the mass-mean truth direction is far more
causally load-bearing than the logistic-regression direction. The folder holds the authors' package
as it was actually run (`codebase/`), the datasets, the executed notebooks and figures, the
per-step run logs, and a summary of the reproduced numbers in `evidence_summary.json`.

### `run/followup/`

Each numbered folder is one self-contained extension study. Inside each you will find the
`instruction.md` that posed the question, the `workspace/` code that answered it, the `results/`
(figures and JSON), a human-readable `report.md`, a machine-readable `result_card.json`, and a
`followup_summary.json`.

- **001-living-update** asks whether the paper's linear truth directions still hold in models
  released in 2024 and 2025, across families (Llama, Qwen, OLMo) and from base to instruction-tuned
  models. They do: mass-mean cross-dataset generalization (`larger_than + smaller_than` to
  `sp_en_trans`) is 0.986 on Llama-3.1-8B, 0.977 on Qwen3-8B-Base, 0.960 on Qwen2.5-7B and 0.935 on
  OLMo-3-7B, against 0.9746 on the replicated 2023 Llama-2-13B anchor, and the direction transfers
  to the instruction-tuned model.

- **002-theory-update** is an analysis-only study (no new GPU work) that reuses 001's artifacts to
  ask how the linear structure scales with parameters, family, generation, and tuning. Truth is
  linearly represented at about 0.4 relative depth in Llama but deeper, about 0.5 to 0.63, in Qwen
  and OLMo; mass-mean cross-dataset generalization is about 0.80 at 7 to 8B and has largely
  saturated near 0.9 by roughly 8 to 13B; and the negation gap collapses from 0.52 on Llama-2-7B to
  0.05 on Qwen3-8B-Base.

- **003-living-update** extends the ladder to 2026 models. The finding still holds on
  Qwen3.5-9B-Base, whose cross-dataset mass-mean generalization reaches 1.00 against the 0.9746
  anchor, but Gemma-4-12B is a partial exception: truth is linearly separable within each dataset
  yet fails to generalize across datasets at the paper's canonical layer.

## A note on the authors' code and data

The files under `run/replication/codebase/` and the dataset CSVs are derived from the authors'
original repository (https://github.com/saprmarks/geometry-of-truth) and remain subject to the
authors' own license and terms. They are included here so the replication and the follow-up studies
are reproducible as run. Please credit Marks and Tegmark for that material.

## Large files

To keep this repository usable on GitHub, the very large artifacts produced during the run (the
residual-stream activation caches, which run to many gigabytes) are not included. Every omission is
listed, with its size and how to regenerate it, in [LARGE_FILES_OMITTED.md](LARGE_FILES_OMITTED.md).
The activations are cheap to recreate: point the authors' `config.ini` at local Llama-2 weights and
rerun `generate_acts.py` as documented in `run/replication/codebase/README.md`.
