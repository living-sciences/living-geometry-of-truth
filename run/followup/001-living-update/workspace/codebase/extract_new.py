"""Extract last-token activations for new-era models over the 8 datasets at the ~6 sweep layers.
Reads capped CSVs (N<=1000, seed 0) from workspace/data/datasets_capped. Saves bf16 tensors to
<acts_root>/<model>/<dataset>/layer_<L>_<idx>.pt (utils.collect_acts layout, batch 25).
Runs a per-arch smoke test first (single statement at probe layer) and aborts on failure.
"""
import os, sys, argparse, configparser, time
import torch as t
import pandas as pd
from fu_common import load_model, get_acts_last

DATASETS = ['cities', 'neg_cities', 'sp_en_trans', 'neg_sp_en_trans',
            'larger_than', 'smaller_than', 'likely', 'cities_cities_conj']

SWEEP = {
    'llama-3.1-8b': [8, 12, 13, 16, 20, 24],
    'qwen2.5-7b': [7, 10, 11, 14, 18, 21],
    'qwen3-8b-base': [9, 14, 18, 22, 27],
    'olmo-3-7b': [8, 12, 13, 16, 20, 24],
    'qwen2.5-7b-instruct': [7, 10, 11, 14, 18, 21],
}

CAPPED = '../data/datasets_capped'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--acts_root', required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    cfg = configparser.ConfigParser(); cfg.read('config.ini')
    probe_layer = eval(cfg[args.model]['probe_layer'])
    layers = SWEEP[args.model]
    assert probe_layer in layers, (probe_layer, layers)

    t.set_grad_enabled(False)
    model = load_model(args.model, args.device)

    # ---- smoke test ----
    sm = get_acts_last(["The city of Paris is in France."], model, [probe_layer])
    shape = tuple(sm[probe_layer].shape)
    print(f'[smoke] {args.model} layer {probe_layer} -> {shape} {sm[probe_layer].dtype}', flush=True)
    assert len(shape) == 2 and shape[0] == 1, f'unexpected smoke shape {shape}'

    # ---- extract ----
    for dataset in DATASETS:
        statements = pd.read_csv(f'{CAPPED}/{dataset}.csv')['statement'].tolist()
        save_dir = os.path.join(args.acts_root, args.model, dataset)
        os.makedirs(save_dir, exist_ok=True)
        t0 = time.time()
        for idx in range(0, len(statements), 25):
            acts = get_acts_last(statements[idx:idx + 25], model, layers)
            for layer, act in acts.items():
                t.save(act.to(t.bfloat16).cpu(), f'{save_dir}/layer_{layer}_{idx}.pt')
        print(f'  {dataset}: {len(statements)} stmts x {len(layers)} layers in {time.time()-t0:.1f}s', flush=True)
    print(f'DONE {args.model}', flush=True)


if __name__ == '__main__':
    main()
