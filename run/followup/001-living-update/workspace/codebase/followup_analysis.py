"""Followup probe/generalization analysis across the model ladder.

Reuses probes.py verbatim (LRProbe/MMProbe/CCSProbe). Mirrors generalization_run.py's
medley/oracle/split logic (split=0.8, seed=42, center=True), but reads activations and labels
via explicit (acts_root, datasets_dir) so llama-2 anchors use the full replication data and new
models use the capped (N<=1000) data. Evaluates on the common 8-dataset grid so the C19 mean
off-diagonal is apples-to-apples across models.

For each model, computes the full generalization matrix at every sweep layer, selects the
validation-best layer (max MM mean-off-diagonal), and records results at BOTH the best-val layer
and the fixed 0.4-relative-depth probe layer. Writes results/generalization_followup.json.
"""
import os, json, argparse
import torch as t
import pandas as pd
from fu_common import collect_acts_dir, load_labels
from probes import LRProbe, MMProbe, CCSProbe

DEVICE = 'cuda:0' if t.cuda.is_available() else 'cpu'
SPLIT = 0.8
SEED = 42

# common evaluation grid (the 8 datasets extracted for new models)
VAL_DATASETS = ['cities', 'neg_cities', 'larger_than', 'smaller_than',
                'sp_en_trans', 'neg_sp_en_trans', 'likely', 'cities_cities_conj']
TRAIN_MEDLEYS = [['cities'], ['cities', 'neg_cities'], ['larger_than'],
                 ['larger_than', 'smaller_than'], ['likely']]
CCS_MEDLEYS = [['cities', 'neg_cities'], ['larger_than', 'smaller_than']]

REPL = '/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/replication/codebase'
FU = '/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/followup/001-living-update'

# model -> (acts_root, datasets_dir, sweep_layers, fixed_probe_layer)
NEW_SWEEP = {
    'llama-3.1-8b': [8, 12, 13, 16, 20, 24],
    'qwen2.5-7b': [7, 10, 11, 14, 18, 21],
    'qwen3-8b-base': [9, 14, 18, 22, 27],
    'olmo-3-7b': [8, 12, 13, 16, 20, 24],
    'qwen2.5-7b-instruct': [7, 10, 11, 14, 18, 21],
}
NEW_PROBE = {'llama-3.1-8b': 13, 'qwen2.5-7b': 11, 'qwen3-8b-base': 14,
             'olmo-3-7b': 13, 'qwen2.5-7b-instruct': 11}


def build_models():
    caps = f'{FU}/workspace/data/datasets_capped'
    full = f'{REPL}/datasets'
    M = {}
    # anchors: full data, all layers available; sweep around the fixed layer for best-val
    M['llama-2-7b'] = (f'{REPL}/acts', full, [7, 10, 13, 16, 20, 24], 13)
    M['llama-2-13b'] = (f'{REPL}/acts', full, [10, 14, 15, 20, 25, 30], 14)
    for m in NEW_SWEEP:
        M[m] = (f'{FU}/acts', caps, NEW_SWEEP[m], NEW_PROBE[m])
    return M


def to_str(l):
    return '+'.join(l)


def load_split(acts_root, datasets_dir, model, dataset, layer):
    acts = collect_acts_dir(acts_root, model, dataset, layer, center=True, device=DEVICE)
    labels = load_labels(datasets_dir, dataset, device=DEVICE)
    assert len(acts) == len(labels), f'{model}/{dataset} acts {len(acts)} != labels {len(labels)}'
    t.manual_seed(SEED)
    train_mask = t.randperm(len(labels)) < int(SPLIT * len(labels))
    return {'train': (acts[train_mask], labels[train_mask]),
            'val': (acts[~train_mask], labels[~train_mask]),
            'full': (acts, labels)}


def matrix_at_layer(acts_root, datasets_dir, model, layer):
    data = {d: load_split(acts_root, datasets_dir, model, d, layer) for d in VAL_DATASETS}

    accs = {'LRProbe': {}, 'MMProbe': {}}
    for ProbeClass in (LRProbe, MMProbe):
        name = ProbeClass.__str__()
        for medley in TRAIN_MEDLEYS:
            tr_acts = t.cat([data[d]['train'][0] for d in medley])
            tr_lab = t.cat([data[d]['train'][1] for d in medley])
            probe = ProbeClass.from_data(tr_acts, tr_lab, device=DEVICE)
            cell = {}
            for vd in VAL_DATASETS:
                if vd in medley:
                    a, lb = data[vd]['val']
                    cell[vd] = (probe.pred(a, iid=True) == lb).float().mean().item()
                else:
                    a, lb = data[vd]['full']
                    cell[vd] = (probe.pred(a, iid=False) == lb).float().mean().item()
            accs[name][to_str(medley)] = cell

    ccs = {}
    for medley in CCS_MEDLEYS:
        a0, l0 = data[medley[0]]['train']
        a1, _ = data[medley[1]]['train']
        probe = CCSProbe.from_data(a0, a1, l0, device=DEVICE)
        cell = {}
        for vd in VAL_DATASETS:
            a, lb = (data[vd]['val'] if vd in medley else data[vd]['full'])
            cell[vd] = (probe.pred(a) == lb).float().mean().item()
        ccs[to_str(medley)] = cell

    oracle = {'LRProbe': {}, 'MMProbe': {}}
    for ProbeClass in (LRProbe, MMProbe):
        name = ProbeClass.__str__()
        for d in VAL_DATASETS:
            a, lb = data[d]['train']
            probe = ProbeClass.from_data(a, lb, device=DEVICE)
            va, vlb = data[d]['val']
            oracle[name][d] = (probe(va, iid=True).round() == vlb).float().mean().item()
    return {'LR': accs['LRProbe'], 'MM': accs['MMProbe'], 'CCS': ccs, 'oracle': oracle}


def mean_offdiag(matrix, probe='MM'):
    src = matrix[probe]
    vals = []
    for medley, cell in src.items():
        mset = medley.split('+')
        for vd, acc in cell.items():
            if vd not in mset:
                vals.append(acc)
    return sum(vals) / len(vals)


def summarize(matrix):
    return {
        'C3_larger_smaller_to_sp_en_trans': {
            'LR': matrix['LR']['larger_than+smaller_than']['sp_en_trans'],
            'MM': matrix['MM']['larger_than+smaller_than']['sp_en_trans'],
            'CCS': matrix['CCS']['larger_than+smaller_than']['sp_en_trans'],
        },
        'C19_mean_offdiag': {'MM': mean_offdiag(matrix, 'MM'), 'LR': mean_offdiag(matrix, 'LR')},
        'oracle_avg': {
            'LR': sum(matrix['oracle']['LRProbe'].values()) / len(matrix['oracle']['LRProbe']),
            'MM': sum(matrix['oracle']['MMProbe'].values()) / len(matrix['oracle']['MMProbe']),
        },
        'matrix': matrix,
    }


def main():
    t.set_grad_enabled(True)
    M = build_models()
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(M.keys()))
    ap.add_argument('--out', default=f'{FU}/results/generalization_followup.json')
    args = ap.parse_args()

    results = {}
    for model in args.models:
        acts_root, datasets_dir, sweep, probe_layer = M[model]
        print(f'\n===== {model} (probe_layer {probe_layer}, sweep {sweep}) =====', flush=True)
        per_layer = {}
        for L in sweep:
            mat = matrix_at_layer(acts_root, datasets_dir, model, L)
            per_layer[L] = mat
            print(f'  layer {L}: MM off-diag {mean_offdiag(mat,"MM"):.4f}  '
                  f'C3(MM)={mat["MM"]["larger_than+smaller_than"]["sp_en_trans"]:.4f}', flush=True)
        best_layer = max(sweep, key=lambda L: mean_offdiag(per_layer[L], 'MM'))
        results[model] = {
            'probe_layer': probe_layer,
            'best_val_layer': best_layer,
            'sweep_layers': sweep,
            'at_probe_layer': summarize(per_layer[probe_layer]),
            'at_best_layer': summarize(per_layer[best_layer]),
            'sweep_mm_offdiag': {L: mean_offdiag(per_layer[L], 'MM') for L in sweep},
        }
        print(f'  -> best_val_layer {best_layer}; '
              f'C3(MM)@probe={results[model]["at_probe_layer"]["C3_larger_smaller_to_sp_en_trans"]["MM"]:.4f}  '
              f'C3(MM)@best={results[model]["at_best_layer"]["C3_larger_smaller_to_sp_en_trans"]["MM"]:.4f}', flush=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
