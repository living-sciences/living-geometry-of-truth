"""Generic last-token residual-stream extractor for a 2026 MM-arch model via plain
transformers output_hidden_states (nnsight cannot wrap these). Auto-discovers the text
decoder ModuleList (len == text_config.num_hidden_layers), computes the sweep from
n_layers, smoke-tests one forward + one activation read, then extracts the 8 datasets.
Layer L file stores output of decoder layer L == hidden_states[L+1][:, -1, :] (left padding).
"""
import os, sys, argparse, time
import torch
import pandas as pd
from transformers import AutoModelForImageTextToText, AutoTokenizer

DATASETS = ['cities', 'neg_cities', 'sp_en_trans', 'neg_sp_en_trans',
            'larger_than', 'smaller_than', 'likely', 'cities_cities_conj']
CAPPED = '../data/datasets_capped'
BATCH = 25


def sweep_for(n):
    fr = [0.25, 0.375, 0.4, 0.5, 0.625, 0.75]
    layers = sorted(set(round(f * n) for f in fr))
    probe = round(0.4 * n)
    return layers, probe


def find_layers(m):
    n = m.config.text_config.num_hidden_layers
    for name, mod in m.named_modules():
        if name.endswith('layers') and hasattr(mod, '__len__') and len(mod) == n:
            return mod, n
    raise RuntimeError('decoder .layers not found')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
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
    layers, n = find_layers(m)
    sweep, probe = sweep_for(n)
    print(f'model_type {m.config.model_type}  n_layers {n}  hidden {m.config.text_config.hidden_size}', flush=True)
    print(f'sweep {sweep}  probe_layer(0.4-depth) {probe}', flush=True)

    ids = tok(['The city of Paris is in France.'], return_tensors='pt', padding=True).to(m.device)
    hs = m(**ids, output_hidden_states=True).hidden_states
    assert len(hs) == n + 1, (len(hs), n)
    print(f'[smoke] hidden_states len {len(hs)}; layer {probe} act '
          f'{tuple(hs[probe+1][:, -1, :].shape)} {hs[probe+1].dtype}', flush=True)

    for dataset in DATASETS:
        statements = pd.read_csv(f'{CAPPED}/{dataset}.csv')['statement'].tolist()
        save_dir = os.path.join(args.acts_root, args.model, dataset)
        os.makedirs(save_dir, exist_ok=True)
        t0 = time.time()
        for idx in range(0, len(statements), BATCH):
            batch = statements[idx:idx + BATCH]
            ids = tok(batch, return_tensors='pt', padding=True).to(m.device)
            hs = m(**ids, output_hidden_states=True).hidden_states
            for L in sweep:
                torch.save(hs[L + 1][:, -1, :].to(torch.bfloat16).cpu(), f'{save_dir}/layer_{L}_{idx}.pt')
        print(f'  {dataset}: {len(statements)} stmts x {len(sweep)} layers in {time.time()-t0:.1f}s', flush=True)
    print(f'DONE {args.model}  sweep={sweep} probe={probe}', flush=True)


if __name__ == '__main__':
    main()
