"""Compute the probe / cross-dataset generalization matrix for the NEW 2026 model
(Qwen3.5-9B-Base) reusing followup_analysis.py's medley/oracle/split logic verbatim
(split=0.8, seed=42, center=True, common 8-dataset grid). Old models are NOT recomputed:
their results are read from the prior 001 living-update on disk and merged, so every old
number is DONE-from-disk and only the 2026 model is freshly computed this session.
Writes results/generalization_followup.json (all models, old + new).
"""
import os, json
import torch as t
from followup_analysis import matrix_at_layer, summarize, mean_offdiag

FU3 = '/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/followup/003-living-update'
PRIOR = f'{FU3}/results/generalization_followup_001.json'   # copied from 001 (old models, DONE)
OUT = f'{FU3}/results/generalization_followup.json'

ACTS_ROOT = f'{FU3}/acts'
CAPPED = f'{FU3}/workspace/data/datasets_capped'

# new 2026 models: (sweep layers, fixed 0.4-relative-depth probe layer)
NEW_2026 = {
    'qwen3.5-9b-base': ([8, 12, 13, 16, 20, 24], 13),      # 32 layers, round(0.4*32)=13
    'gemma-4-12b':     ([12, 18, 19, 24, 30, 36], 19),     # 48 layers, round(0.4*48)=19
}


def main():
    t.set_grad_enabled(True)
    prior = json.load(open(PRIOR))
    print(f'reused old models from disk: {list(prior.keys())}', flush=True)

    for MODEL, (SWEEP, PROBE_LAYER) in NEW_2026.items():
        print(f'\n===== {MODEL} (sweep {SWEEP}, probe {PROBE_LAYER}) =====', flush=True)
        per_layer = {}
        for L in SWEEP:
            mat = matrix_at_layer(ACTS_ROOT, CAPPED, MODEL, L)
            per_layer[L] = mat
            print(f'  layer {L}: MM off-diag {mean_offdiag(mat,"MM"):.4f}  '
                  f'C3(MM)={mat["MM"]["larger_than+smaller_than"]["sp_en_trans"]:.4f}', flush=True)
        best_layer = max(SWEEP, key=lambda L: mean_offdiag(per_layer[L], 'MM'))
        prior[MODEL] = {
            'probe_layer': PROBE_LAYER,
            'best_val_layer': best_layer,
            'sweep_layers': SWEEP,
            'at_probe_layer': summarize(per_layer[PROBE_LAYER]),
            'at_best_layer': summarize(per_layer[best_layer]),
            'sweep_mm_offdiag': {L: mean_offdiag(per_layer[L], 'MM') for L in SWEEP},
        }
        p = prior[MODEL]
        print(f'  {MODEL}: best_val_layer {best_layer}', flush=True)
        print(f'    C3(MM)@probe(L{PROBE_LAYER})={p["at_probe_layer"]["C3_larger_smaller_to_sp_en_trans"]["MM"]:.4f}  '
              f'C3(MM)@best(L{best_layer})={p["at_best_layer"]["C3_larger_smaller_to_sp_en_trans"]["MM"]:.4f}', flush=True)
        print(f'    C3 LR@best={p["at_best_layer"]["C3_larger_smaller_to_sp_en_trans"]["LR"]:.4f}  '
              f'CCS@best={p["at_best_layer"]["C3_larger_smaller_to_sp_en_trans"]["CCS"]:.4f}', flush=True)
        print(f'    C19 MM off-diag@best={p["at_best_layer"]["C19_mean_offdiag"]["MM"]:.4f}  '
              f'oracle MM={p["at_best_layer"]["oracle_avg"]["MM"]:.4f} LR={p["at_best_layer"]["oracle_avg"]["LR"]:.4f}', flush=True)

    with open(OUT, 'w') as f:
        json.dump(prior, f, indent=2)
    print(f'\nwrote {OUT} ({len(prior)} models)')


if __name__ == '__main__':
    main()
