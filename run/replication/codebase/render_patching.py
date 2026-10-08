"""Render patching heatmaps (Token x Layer, patched TRUE-FALSE logit diff) for every record
in experimental_outputs/patching_results.json. Ported from patching.ipynb, which loaded a full
meta-llama/Llama-2-70b-hf model just for its tokenizer (gated + unavailable). Here we load only
the local tokenizer for token labels and save one PNG per record to figures/.
"""
import json
import configparser
import numpy as np
import plotly.express as px
from transformers import AutoTokenizer

config = configparser.ConfigParser()
config.read('config.ini')

with open('experimental_outputs/patching_results.json', 'r') as f:
    records = json.load(f)

_tok_cache = {}
def get_tokenizer(model_name):
    if model_name not in _tok_cache:
        _tok_cache[model_name] = AutoTokenizer.from_pretrained(config[model_name]['weights_directory'])
    return _tok_cache[model_name]

for out in records:
    false_prompt = out['false_prompt']
    logit_diffs = out['logit_diffs']
    n_toks = len(logit_diffs)
    model_name = out['model']
    prompt_kind = 'cities' if 'city of' in false_prompt else ('larger_than' if 'larger than' in false_prompt else 'sp_en_trans')

    # transpose to [layer][token] with tokens in reading order (notebook's transform)
    ld_T = [[logit_diffs[i][j] for i in range(len(logit_diffs))[::-1]] for j in range(len(logit_diffs[0]))]

    tok = get_tokenizer(model_name)
    token_ids = tok(false_prompt)['input_ids']
    tokens = [tok.decode([tid]) + f' ({idx})' for idx, tid in enumerate(token_ids)][-n_toks:]

    fig = px.imshow(
        ld_T,
        x=tokens,
        labels=dict(x='Token', y='Layer', color='TRUE-FALSE logit diff'),
        color_continuous_scale='blues',
        aspect='auto',
    )
    fig.update_layout(title=f'Activation patching: {model_name} ({prompt_kind} prompt)',
                      width=900, height=800)
    path = f'figures/patching_{prompt_kind}_{model_name}.png'
    fig.write_image(path, scale=2)
    print(f'saved {path}  (n_toks={n_toks}, n_layers={len(logit_diffs[0])})')

print('done')
