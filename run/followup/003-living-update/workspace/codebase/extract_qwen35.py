"""Extract last-token residual-stream activations for the 2026 MM-arch model
(Qwen3.5-9B-Base) via plain transformers output_hidden_states (nnsight cannot wrap qwen3_5).

Convention match with the nnsight pipeline used for the 2024/2025 models:
  nnsight `model.model.layers[L].output[:, -1, :]` == output of decoder layer L
  == transformers hidden_states[L+1][:, -1, :]  (hidden_states[0] is embeddings).
Verified exact (diff 0.0) in smoke_qwen35.py.

Left-padding + attention_mask so [:, -1, :] is always the true last content token.
Saves bf16 tensors to <acts_root>/<model>/<dataset>/layer_<L>_<idx>.pt (batch 25),
the layout fu_common.collect_acts_dir / utils.collect_acts expect.
"""
import os, sys, argparse, time
import torch
import pandas as pd
from transformers import AutoModelForImageTextToText, AutoTokenizer

DATASETS = ['cities', 'neg_cities', 'sp_en_trans', 'neg_sp_en_trans',
            'larger_than', 'smaller_than', 'likely', 'cities_cities_conj']
SWEEP = [8, 12, 13, 16, 20, 24]           # {0.25,0.375,0.4,0.5,0.625,0.75}*32
PROBE_LAYER = 13                          # round(0.4*32)
CAPPED = '../data/datasets_capped'
BATCH = 25


def find_layers(m):
    """Return the ModuleList of decoder layers (the residual stream)."""
    n = m.config.text_config.num_hidden_layers
    for name, mod in m.named_modules():
        if name.endswith('layers') and hasattr(mod, '__len__') and len(mod) == n:
            return mod
    raise RuntimeError('decoder .layers not found')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3.5-9b-base')
    ap.add_argument('--path', required=True)
    ap.add_argument('--acts_root', required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    print(f'loading {args.path}', flush=True)
    m = AutoModelForImageTextToText.from_pretrained(args.path, dtype=torch.bfloat16, device_map='auto').eval()
    tok = AutoTokenizer.from_pretrained(args.path)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    layers = find_layers(m)
    n_layers = len(layers)
    assert PROBE_LAYER in SWEEP and max(SWEEP) < n_layers, (SWEEP, n_layers)

    # smoke: single statement at probe layer
    ids = tok(['The city of Paris is in France.'], return_tensors='pt', padding=True).to(m.device)
    hs = m(**ids, output_hidden_states=True).hidden_states
    assert len(hs) == n_layers + 1, (len(hs), n_layers)
    print(f'[smoke] hidden_states len {len(hs)}; layer {PROBE_LAYER} act '
          f'{tuple(hs[PROBE_LAYER+1][:, -1, :].shape)} {hs[PROBE_LAYER+1].dtype}', flush=True)

    for dataset in DATASETS:
        statements = pd.read_csv(f'{CAPPED}/{dataset}.csv')['statement'].tolist()
        save_dir = os.path.join(args.acts_root, args.model, dataset)
        os.makedirs(save_dir, exist_ok=True)
        t0 = time.time()
        for idx in range(0, len(statements), BATCH):
            batch = statements[idx:idx + BATCH]
            ids = tok(batch, return_tensors='pt', padding=True).to(m.device)
            hs = m(**ids, output_hidden_states=True).hidden_states
            for L in SWEEP:
                act = hs[L + 1][:, -1, :].to(torch.bfloat16).cpu()
                torch.save(act, f'{save_dir}/layer_{L}_{idx}.pt')
        print(f'  {dataset}: {len(statements)} stmts x {len(SWEEP)} layers in {time.time()-t0:.1f}s', flush=True)
    print(f'DONE {args.model}', flush=True)


if __name__ == '__main__':
    main()
