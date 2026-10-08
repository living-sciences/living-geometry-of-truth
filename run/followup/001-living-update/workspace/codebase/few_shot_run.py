"""Calibrated 5-shot baseline (Step 5), ported from few_shot.py to the nnsight 0.2.21
trace API. few_shot.py used the pre-0.2 `model.forward(remote=...) as runner: runner.invoke(...)`
+ `runner.output['past_key_values']` KV-cache API, which nnsight 0.2.21 does not provide
(model.forward proxies straight to LlamaForCausalLM.forward -> TypeError on `remote`).

Methodology preserved exactly: build an n_shots few-shot prompt ('statement TRUE/FALSE\\n'),
then for each held-out query read the next-token logits after `prompt + statement`, take
diff = P(' TRUE') - P(' FALSE'); if calibrated, gamma = median(diffs) and predict diff > gamma.
The only change vs few_shot.py is recomputing the prompt prefix per batch (no KV-cache reuse),
which yields identical logits. Output JSON schema matches few_shot.py.
"""
import torch as t
import pandas as pd
import os
import json
import argparse
from tqdm import tqdm
from generate_acts import load_model

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 42  # fix the shot sampling for reproducibility (setup value, not a result)


def get_few_shot_accuracy(datasets, model, n_shots=5, batch_size=32, calibrated=True):
    model.tokenizer.padding_side = 'left'  # last token at index -1 for each row
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    true_idx = model.tokenizer.encode(' TRUE')[-1]
    false_idx = model.tokenizer.encode(' FALSE')[-1]

    outs = []
    for dataset in datasets:
        out = {'dataset': dataset, 'n_shots': n_shots}
        df = pd.read_csv(os.path.join(ROOT, 'datasets', f"{dataset}.csv"))
        shots = df.sample(n_shots, random_state=SEED)
        queries = df.drop(shots.index)

        prompt = ''
        for _, shot in shots.iterrows():
            prompt += f'{shot["statement"]} '
            prompt += 'TRUE\n' if bool(shot['label']) else 'FALSE\n'
        out['shots'] = shots['statement'].tolist()
        out['prompt'] = prompt

        diffs = []
        stmts = queries['statement'].tolist()
        for b in tqdm(range(0, len(stmts), batch_size), desc=f'{dataset}'):
            batch = [prompt + s for s in stmts[b:b + batch_size]]
            with model.trace(batch, scan=False, validate=False):
                last = model.lm_head.output[:, -1, :].save()
            probs = last.value.float().softmax(-1)
            diffs.append((probs[:, true_idx] - probs[:, false_idx]).cpu())
        diffs = t.cat(diffs)

        if calibrated:
            gamma = t.sort(diffs).values[len(diffs) // 2]
            out['gamma'] = gamma.item()
        else:
            gamma = t.tensor(0.0)

        predicted = diffs > gamma
        ground_truth = t.tensor(queries['label'].values).bool()
        out['acc'] = (predicted == ground_truth).float().mean().item()
        outs.append(out)
        print(f'  {dataset}: acc={out["acc"]:.4f} gamma={out.get("gamma", 0):.4g} (n_query={len(stmts)})')
    return outs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+')
    parser.add_argument('--model', type=str, default='llama-2-13b')
    parser.add_argument('--n_shots', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--uncalibrated', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    t.set_grad_enabled(False)
    model = load_model(args.model, device=args.device)
    outs = get_few_shot_accuracy(args.datasets, model, args.n_shots, args.batch_size, not args.uncalibrated)
    for out in outs:
        out['model'] = args.model

    path = os.path.join(ROOT, 'experimental_outputs', 'few_shot_results.json')
    with open(path, 'r') as f:
        data = json.load(f)
    data.extend(outs)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print('appended', len(outs), 'results to', path)
