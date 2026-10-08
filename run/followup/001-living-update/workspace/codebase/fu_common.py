"""Followup-common helpers. Reuses generate_acts.load_model and probes.py verbatim.

Difference from the replication's generate_acts.get_acts: transformers >=4.57 decoder layers
return the hidden-state TENSOR directly (shape [batch, seq, hidden]) rather than a tuple, so
the last-token selector is `.output[:, -1, :]` (verified for Llama-3.1/Qwen2.5/Qwen3/OLMo-3),
not `.output[0][:, -1, :]`. Everything else (probes, medleys, centering) is identical.
"""
import os
import torch as t
import pandas as pd
from glob import glob
from generate_acts import load_model  # noqa: F401  (re-exported)

ACTS_BATCH_SIZE = 25


def get_acts_last(statements, model, layers):
    """Last-token activations at given layers, local (remote=False). transformers>=4.57 API."""
    acts = {}
    with model.trace(statements, scan=False, validate=False):
        for layer in layers:
            acts[layer] = model.model.layers[layer].output[:, -1, :].save()
    return {layer: act.value for layer, act in acts.items()}


def collect_acts_dir(acts_root, model, dataset, layer, center=True, device='cpu'):
    """Mirror of utils.collect_acts but with an explicit acts_root (so followup new-model
    activations under followup/.../acts can be read without touching the replication tree)."""
    directory = os.path.join(acts_root, model, dataset)
    files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
    if len(files) == 0:
        raise ValueError(f"acts for {model}/{dataset} layer {layer} not found in {directory}")
    n = ACTS_BATCH_SIZE * len(files)
    acts = [t.load(os.path.join(directory, f'layer_{layer}_{i}.pt')).to(device)
            for i in range(0, n, ACTS_BATCH_SIZE)]
    acts = t.cat(acts, dim=0).float().to(device)
    if center:
        acts = acts - t.mean(acts, dim=0)
    return acts


def load_labels(datasets_dir, dataset, device='cpu'):
    df = pd.read_csv(os.path.join(datasets_dir, f'{dataset}.csv'))
    return t.Tensor(df['label'].values).to(device)
