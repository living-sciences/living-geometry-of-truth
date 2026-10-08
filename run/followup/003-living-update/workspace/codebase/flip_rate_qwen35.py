"""Cheap causal analog of C2/C17 for the 2026 MM-arch model (Qwen3.5-9B-Base), using
transformers forward HOOKS (nnsight cannot wrap qwen3_5). Mirrors flip_rate.py's recipe:
extract the MM truth direction from the model's own cities+neg_cities acts at the probe layer,
scale it by the true/false mean separation, then during a forward pass over the sp_en_trans
few-shot prompt add it to false statements (push FALSE->TRUE) and subtract from true statements
(push TRUE->FALSE) at hidden states from intervene_layer..probe_layer at the two suffix positions.
Report MM flip-rate = fraction whose predicted TRUE/FALSE label flips to the intended target.
Also sweeps magnitude (1x,2x,4x,8x). Forward passes only.
"""
import os, json, argparse
import torch as t
import pandas as pd
from transformers import AutoModelForImageTextToText, AutoTokenizer
from fu_common import collect_acts_dir
from probes import MMProbe

FU3 = '/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/followup/003-living-update'
CAPPED = f'{FU3}/workspace/data/datasets_capped'
PROBE_LAYER = 13
INTERVENE_LAYER = 7

PROMPT = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
"""


def find_layers(m):
    n = m.config.text_config.num_hidden_layers
    for name, mod in m.named_modules():
        if name.endswith('layers') and hasattr(mod, '__len__') and len(mod) == n:
            return mod
    raise RuntimeError('layers not found')


def prepare(subset):
    df = pd.read_csv(f'{CAPPED}/sp_en_trans.csv')
    stmts = df[df['label'] == (1 if subset == 'true' else 0)]['statement'].tolist()
    return [PROMPT + s + ' This statement is:' for s in stmts if s not in PROMPT]


def pdiffs(m, tok, layers, queries, direction, sign, offsets, positions, true_idx, false_idx, batch_size=32):
    """sign: 0 = no intervention; +1 add; -1 subtract. positions = list of absolute-from-right indices."""
    out = []
    handles = []
    if sign != 0:
        d = (sign * direction)
        def make_hook():
            def hook(mod, inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                for pos in positions:
                    hs[:, pos, :] = hs[:, pos, :] + d
                if isinstance(output, tuple):
                    return (hs,) + tuple(output[1:])
                return hs
            return hook
        for L in layers:
            handles.append(m.model.language_model.layers[L].register_forward_hook(make_hook()))
    try:
        for b in range(0, len(queries), batch_size):
            batch = queries[b:b + batch_size]
            ids = tok(batch, return_tensors='pt', padding=True).to(m.device)
            logits = m(**ids).logits[:, -1, :]
            probs = logits.float().softmax(-1)
            out.append((probs[:, true_idx] - probs[:, false_idx]).cpu())
    finally:
        for h in handles:
            h.remove()
    return t.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3.5-9b-base')
    ap.add_argument('--path', required=True)
    ap.add_argument('--out', default=f'{FU3}/results/flip_rates.json')
    ap.add_argument('--scales', nargs='+', type=float, default=[1.0, 2.0, 4.0, 8.0])
    args = ap.parse_args()

    # MM direction from the model's own cities+neg_cities acts at probe layer
    acts_root = f'{FU3}/acts'
    a = t.cat([collect_acts_dir(acts_root, args.model, 'cities', PROBE_LAYER, center=False),
               collect_acts_dir(acts_root, args.model, 'neg_cities', PROBE_LAYER, center=False)])
    lab = t.cat([t.Tensor(pd.read_csv(f'{CAPPED}/cities.csv')['label'].values),
                 t.Tensor(pd.read_csv(f'{CAPPED}/neg_cities.csv')['label'].values)])
    a = a - a.mean(0)
    with t.enable_grad():
        probe = MMProbe.from_data(a, lab)
    direction = probe.direction
    direction = direction / direction.norm()
    diff = ((a[lab == 1].mean(0) - a[lab == 0].mean(0)) @ direction)
    unit = (diff * direction).detach()   # 1x = natural cities separation magnitude

    t.set_grad_enabled(False)
    print('loading', args.path, flush=True)
    m = AutoModelForImageTextToText.from_pretrained(args.path, dtype=t.bfloat16, device_map='auto').eval()
    tok = AutoTokenizer.from_pretrained(args.path)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _ = find_layers(m)  # assert structure

    true_idx = tok.encode(' TRUE')[-1]
    false_idx = tok.encode(' FALSE')[-1]
    len_suffix = len(tok.encode('This statement is:'))
    # two suffix positions (matches flip_rate.py: offsets 0 and -1 within the suffix, from right)
    positions = [-len_suffix, -len_suffix + 1]
    layers = list(range(INTERVENE_LAYER, PROBE_LAYER + 1))
    q_false, q_true = prepare('false'), prepare('true')

    base_false = pdiffs(m, tok, layers, q_false, unit.to(m.device, t.bfloat16), 0, None, positions, true_idx, false_idx)
    base_true = pdiffs(m, tok, layers, q_true, unit.to(m.device, t.bfloat16), 0, None, positions, true_idx, false_idx)
    n_f = int((base_false < 0).sum()); n_t = int((base_true > 0).sum())

    sweep = {}
    for sc in args.scales:
        d = (sc * unit).to(m.device, t.bfloat16)
        add_false = pdiffs(m, tok, layers, q_false, d, +1, None, positions, true_idx, false_idx)
        sub_true = pdiffs(m, tok, layers, q_true, d, -1, None, positions, true_idx, false_idx)
        flip_f2t = ((base_false < 0) & (add_false > 0)).float().mean().item()
        flip_t2f = ((base_true > 0) & (sub_true < 0)).float().mean().item()
        overall = (((base_false < 0) & (add_false > 0)).sum().item() +
                   ((base_true > 0) & (sub_true < 0)).sum().item()) / max(1, (n_f + n_t))
        sweep[str(sc)] = {
            'flip_rate_false_to_true': flip_f2t, 'flip_rate_true_to_false': flip_t2f,
            'flip_rate_overall_on_correct': overall,
            'mean_pdiff_false_add': add_false.mean().item(), 'mean_pdiff_true_sub': sub_true.mean().item(),
        }
        print(f'  scale {sc}: overall flip {overall:.3f} (f2t {flip_f2t:.3f}, t2f {flip_t2f:.3f}) '
              f'mean p_diff true->sub {sub_true.mean().item():+.3f}', flush=True)

    rec = {
        'model': args.model, 'probe': 'MM', 'backend': 'transformers-hooks (nnsight cannot wrap qwen3_5)',
        'train_direction': 'cities+neg_cities', 'val_dataset': 'sp_en_trans',
        'probe_layer': PROBE_LAYER, 'intervene_layers': [INTERVENE_LAYER, PROBE_LAYER],
        'n_false': len(q_false), 'n_true': len(q_true),
        'n_false_correct_baseline': n_f, 'n_true_correct_baseline': n_t,
        'mean_pdiff_false_base': base_false.mean().item(), 'mean_pdiff_true_base': base_true.mean().item(),
        'flip_rate_overall_on_correct': sweep['1.0']['flip_rate_overall_on_correct'],
        'flip_rate_false_to_true': sweep['1.0']['flip_rate_false_to_true'],
        'flip_rate_true_to_false': sweep['1.0']['flip_rate_true_to_false'],
        'magnitude_sweep': sweep, 'scale': 1.0,
    }
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in rec.items() if k != 'magnitude_sweep'}, indent=2))
    allr = {}
    if os.path.exists(args.out):
        allr = json.load(open(args.out))
    allr[args.model] = rec
    with open(args.out, 'w') as f:
        json.dump(allr, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
