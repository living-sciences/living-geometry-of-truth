"""Cheap causal analog of C2/C17 (a single MM direction, NOT the full Table-2 sweep).

Extract the mass-mean truth direction from the model's own cities+neg_cities activations at the
probe layer, scale it by the true/false mean separation (as interventions.py does), then during a
forward pass over the sp_en_trans few-shot prompt add it to false statements (push FALSE->TRUE) and
subtract it from true statements (push TRUE->FALSE), across hidden states from intervene_layer..probe
layer at the two suffix positions. Report MM flip-rate = fraction of statements whose predicted
TRUE/FALSE label flips to the intended target. Forward passes only.
"""
import os, json, argparse, configparser
import torch as t
import pandas as pd
from fu_common import load_model, collect_acts_dir
from probes import MMProbe

FU = '/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/followup/001-living-update'
CAPPED = '../data/datasets_capped'

PROMPT = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
"""


def prepare(subset):
    df = pd.read_csv(f'{CAPPED}/sp_en_trans.csv')
    stmts = df[df['label'] == (1 if subset == 'true' else 0)]['statement'].tolist()
    return [PROMPT + s + ' This statement is:' for s in stmts if s not in PROMPT]


def pdiffs(model, queries, direction, hidden_states, intervention, batch_size=32):
    true_idx = model.tokenizer.encode(' TRUE')[-1]
    false_idx = model.tokenizer.encode(' FALSE')[-1]
    len_suffix = len(model.tokenizer.encode('This statement is:'))
    out = []
    for b in range(0, len(queries), batch_size):
        batch = queries[b:b + batch_size]
        with model.trace(batch, scan=False, validate=False):
            if intervention != 'none':
                for layer, offset in hidden_states:
                    hs = model.model.layers[layer].output
                    d = direction if intervention == 'add' else -direction
                    hs[:, -len_suffix + offset, :] = hs[:, -len_suffix + offset, :] + d
            logits = model.lm_head.output[:, -1, :]
            probs = logits.softmax(-1)
            out.append((probs[:, true_idx] - probs[:, false_idx]).save())
    return t.cat([o.value for o in out]).float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default=f'{FU}/results/flip_rates.json')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--no_save', action='store_true')
    args = ap.parse_args()

    cfg = configparser.ConfigParser(); cfg.read('config.ini')
    probe_layer = eval(cfg[args.model]['probe_layer'])
    intervene_layer = eval(cfg[args.model]['intervene_layer'])
    hidden_states = []
    for L in range(intervene_layer, probe_layer + 1):
        hidden_states += [(L, -1), (L, 0)]

    # MM direction from the model's own cities+neg_cities acts at probe layer
    acts_root = f'{FU}/acts'
    a = t.cat([collect_acts_dir(acts_root, args.model, 'cities', probe_layer, center=False),
               collect_acts_dir(acts_root, args.model, 'neg_cities', probe_layer, center=False)])
    lab = t.cat([t.Tensor(pd.read_csv(f'{CAPPED}/cities.csv')['label'].values),
                 t.Tensor(pd.read_csv(f'{CAPPED}/neg_cities.csv')['label'].values)])
    a = a - a.mean(0)
    with t.enable_grad():
        probe = MMProbe.from_data(a, lab)
    direction = probe.direction
    direction = direction / direction.norm()
    diff = ((a[lab == 1].mean(0) - a[lab == 0].mean(0)) @ direction)
    direction = (args.scale * diff * direction).detach()

    t.set_grad_enabled(False)
    model = load_model(args.model, args.device)
    direction = direction.to(args.device, t.bfloat16)

    q_false, q_true = prepare('false'), prepare('true')
    base_false = pdiffs(model, q_false, direction, hidden_states, 'none')
    add_false = pdiffs(model, q_false, direction, hidden_states, 'add')
    base_true = pdiffs(model, q_true, direction, hidden_states, 'none')
    sub_true = pdiffs(model, q_true, direction, hidden_states, 'subtract')

    # flip = predicted label crosses 0 to the intended target
    flip_f2t = ((base_false < 0) & (add_false > 0)).float().mean().item()   # FALSE->TRUE
    flip_t2f = ((base_true > 0) & (sub_true < 0)).float().mean().item()     # TRUE->FALSE
    n_f = int((base_false < 0).sum()); n_t = int((base_true > 0).sum())
    overall = (((base_false < 0) & (add_false > 0)).sum().item() +
               ((base_true > 0) & (sub_true < 0)).sum().item()) / max(1, (n_f + n_t))

    rec = {
        'model': args.model, 'probe': 'MM', 'train_direction': 'cities+neg_cities',
        'val_dataset': 'sp_en_trans', 'probe_layer': probe_layer,
        'intervene_layers': [intervene_layer, probe_layer],
        'n_false': len(q_false), 'n_true': len(q_true),
        'n_false_correct_baseline': n_f, 'n_true_correct_baseline': n_t,
        'flip_rate_false_to_true': flip_f2t, 'flip_rate_true_to_false': flip_t2f,
        'flip_rate_overall_on_correct': overall,
        'mean_pdiff_false_base': base_false.mean().item(), 'mean_pdiff_false_add': add_false.mean().item(),
        'mean_pdiff_true_base': base_true.mean().item(), 'mean_pdiff_true_sub': sub_true.mean().item(),
    }
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, indent=2))

    rec['scale'] = args.scale
    if args.no_save:
        return
    allr = {}
    if os.path.exists(args.out):
        allr = json.load(open(args.out))
    allr[args.model] = rec
    with open(args.out, 'w') as f:
        json.dump(allr, f, indent=2)


if __name__ == '__main__':
    main()
