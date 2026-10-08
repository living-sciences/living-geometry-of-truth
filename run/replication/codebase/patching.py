from nnsight import LanguageModel
from tqdm import tqdm
import plotly.express as px
import torch as t
import json
import argparse
from generate_acts import load_model


# Prompt pairs (false_prompt / true_prompt) that differ only in the final query statement,
# so the true/false completion is the only semantic change near the end.
PROMPT_PAIRS = {
    'cities': (
        "The city of Tokyo is in Japan. This statement is: TRUE\nThe city of Hanoi is in Poland. This statement is: FALSE\nThe city of Chicago is in Canada. This statement is:",
        "The city of Tokyo is in Japan. This statement is: TRUE\nThe city of Hanoi is in Poland. This statement is: FALSE\nThe city of Toronto is in Canada. This statement is:",
    ),
    'larger_than': (
        "Fifty-one is larger than seventy-five. This statement is: FALSE\nEighty-two is larger than sixty-four. This statement is: TRUE\nEighty-one is larger than eighty-four. This statement is:",
        "Fifty-one is larger than seventy-five. This statement is: FALSE\nEighty-two is larger than sixty-four. This statement is: TRUE\nEighty-six is larger than eighty-four. This statement is:",
    ),
    'sp_en_trans': (
        """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'diccionario' means 'dictionary'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
The Spanish word 'con' means 'one'. This statement is:""",
        """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'diccionario' means 'dictionary'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
The Spanish word 'uno' means 'one'. This statement is:""",
    ),
}


def patching_experiment(model_name, continuation_idx=None, device='remote', prompt='sp_en_trans'):

    t.set_grad_enabled(False)
    model = load_model(model_name, device=device)
    layers = model.model.layers
    remote = device == 'remote'

    false_prompt, true_prompt = PROMPT_PAIRS[prompt]

    # check that prompts have the same length
    false_toks = model.tokenizer(false_prompt).input_ids
    true_toks = model.tokenizer(true_prompt).input_ids
    if len(false_toks) != len(true_toks):
        raise ValueError(f"False prompt has length {len(false_toks)} but true prompt has length {len(true_toks)}")

    # find number of tokens after the change
    sames = [false_tok == true_tok for false_tok, true_tok in zip(false_toks, true_toks)]
    n_toks = sames[::-1].index(False) + 1

    # nnsight 0.2.21 trace API (ported from the pre-0.2 forward/runner API)
    with model.trace(true_prompt, scan=False, validate=False):
        true_acts = [layer.output[0].save() for layer in model.model.layers]
    true_acts = [act.value for act in true_acts]

    if continuation_idx is not None: # if picking up an experiment that failed
        with open('experimental_outputs/patching_results.json', 'r') as f:
            outs = json.load(f)
        out = outs[continuation_idx]
        assert out['model'] == model_name
        assert out['false_prompt'] == false_prompt
        assert out['true_prompt'] == true_prompt
        logit_diffs = out['logit_diffs']
    else:
        out = {
            'model' : model_name,
            'false_prompt' : false_prompt,
            'true_prompt' : true_prompt,
        }
        logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]
        out['logit_diffs'] = logit_diffs
        with open('experimental_outputs/patching_results.json', 'r') as f:
            outs = json.load(f)
        outs.append(out)
        with open('experimental_outputs/patching_results.json', 'w') as f:
            json.dump(outs, f, indent=4)
        continuation_idx = -1

    t_tok = model.tokenizer(" TRUE").input_ids[-1]
    f_tok = model.tokenizer(" FALSE").input_ids[-1]

    for tok_idx in range(1, n_toks + 1):
        for layer_idx, layer in enumerate(model.model.layers):
            if logit_diffs[tok_idx - 1][layer_idx] is not None:
                continue # already computed
            with model.trace(false_prompt, scan=False, validate=False):
                layer.output[0][0,-tok_idx,:] = true_acts[layer_idx][0,-tok_idx,:]
                logits = model.lm_head.output
                logit_diff = logits[0, -1, t_tok] - logits[0, -1, f_tok]
                logit_diff = logit_diff.save()
            logit_diffs[tok_idx - 1][layer_idx] = logit_diff.value.item()
            
            outs[continuation_idx] = out
            with open('experimental_outputs/patching_results.json', 'w') as f:
                json.dump(outs, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-70b')
    parser.add_argument('--continuation_idx', type=int, default=None)
    parser.add_argument('--device', type=str, default='remote')
    parser.add_argument('--prompt', type=str, default='sp_en_trans', choices=list(PROMPT_PAIRS.keys()))
    args = parser.parse_args()

    patching_experiment(args.model, args.continuation_idx, args.device, args.prompt)