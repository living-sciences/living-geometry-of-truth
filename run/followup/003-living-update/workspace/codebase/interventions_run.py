"""Orchestrator for Step 6: reproduce the causal-intervention NIE table (Table 2) for
Llama-2-13B on val_dataset=sp_en_trans. Loads the model ONCE (the CLI reloads it on every
call -> ~48 reloads), trains each (train_set, probe) direction, and runs the 4 passes per
cell: PD- (false,none), PD-* (false,add), PD+ (true,none), PD+* (true,subtract). Appends
each pass to experimental_outputs/label_change_intervention_results.json (same schema as
interventions.py) and writes the derived NIEs to experimental_outputs/nie_table_13b.json.

NIE_false->true = (PD-* - PD-) / (PD+ - PD-)
NIE_true->false = (PD+* - PD+) / (PD- - PD+)
"""
import torch as t
import pandas as pd
import json
import configparser
from utils import collect_acts
from generate_acts import load_model
from probes import LRProbe, MMProbe, CCSProbe
from interventions import intervention_experiment, prepare_data

MODEL = 'llama-2-13b'
VAL = 'sp_en_trans'
DEVICE = 'cuda:0'
RESULTS = 'experimental_outputs/label_change_intervention_results.json'

config = configparser.ConfigParser()
config.read('config.ini')
start_layer = eval(config[MODEL]['intervene_layer'])   # 8
end_layer = eval(config[MODEL]['probe_layer'])         # 14
noperiod = eval(config[MODEL]['noperiod'])             # False

# hidden states to intervene over (interventions.py lines 96-104)
if noperiod:
    hidden_states = [(layer, -1) for layer in range(start_layer, end_layer + 1)]
else:
    hidden_states = []
    for layer in range(start_layer, end_layer + 1):
        hidden_states.append((layer, -1))
        hidden_states.append((layer, 0))

# 13B sp_en_trans few-shot prompt (hardcoded in interventions.py)
PROMPT = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
"""

# Table 2 (13B column) cells
CELLS = [
    (['cities'], 'LRProbe'),
    (['cities'], 'MMProbe'),
    (['cities', 'neg_cities'], 'LRProbe'),
    (['cities', 'neg_cities'], 'MMProbe'),
    (['cities', 'neg_cities'], 'CCSProbe'),
    (['larger_than'], 'LRProbe'),
    (['larger_than'], 'MMProbe'),
    (['larger_than', 'smaller_than'], 'LRProbe'),
    (['larger_than', 'smaller_than'], 'MMProbe'),
    (['larger_than', 'smaller_than'], 'CCSProbe'),
    (['likely'], 'LRProbe'),
    (['likely'], 'MMProbe'),
]


def get_direction(train_datasets, probe_name):
    """Replicates interventions.py lines 106-132: train probe, build a truth direction
    scaled by the true/false mean separation along the (unit) probe direction."""
    ProbeClass = {'LRProbe': LRProbe, 'MMProbe': MMProbe, 'CCSProbe': CCSProbe}[probe_name]
    # probe training needs grad; the intervention forward passes must NOT build an autograd
    # graph over the 13B model (that OOMs the A40), so grad stays globally disabled (set in
    # main) and is enabled only for the training call here.
    with t.enable_grad():
        if ProbeClass in (LRProbe, MMProbe):
            acts, labels = [], []
            for dataset in train_datasets:
                acts.append(collect_acts(dataset, MODEL, end_layer, noperiod=noperiod).to(DEVICE))
                labels.append(t.Tensor(pd.read_csv(f'datasets/{dataset}.csv')['label'].tolist()).to(DEVICE))
            acts, labels = t.cat(acts), t.cat(labels)
            probe = ProbeClass.from_data(acts, labels, device=DEVICE)
        else:  # CCS
            acts = collect_acts(train_datasets[0], MODEL, end_layer, noperiod=noperiod).to(DEVICE)
            neg_acts = collect_acts(train_datasets[1], MODEL, end_layer, noperiod=noperiod).to(DEVICE)
            labels = t.Tensor(pd.read_csv(f'datasets/{train_datasets[0]}.csv')['label'].tolist()).to(DEVICE)
            probe = ProbeClass.from_data(acts, neg_acts, labels=labels, device=DEVICE)

    direction = probe.direction
    true_acts, false_acts = acts[labels == 1], acts[labels == 0]
    true_mean, false_mean = true_acts.mean(0), false_acts.mean(0)
    direction = direction / direction.norm()
    diff = (true_mean - false_mean) @ direction
    direction = diff * direction
    return direction.detach().to(DEVICE, t.bfloat16), ProbeClass.__name__


def append_record(train_datasets, probe_cls_name, p_diff, tot, intervention, subset):
    out = {
        'model': MODEL, 'train_datasets': train_datasets, 'val_dataset': VAL,
        'probe class': probe_cls_name, 'prompt': PROMPT,
        'p_diff': p_diff, 'tot': tot, 'intervention': intervention,
        'subset': subset, 'hidden_states': hidden_states,
    }
    with open(RESULTS, 'r') as f:
        data = json.load(f)
    data.append(out)
    with open(RESULTS, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    t.set_grad_enabled(False)  # intervention forward passes run under no-grad; see get_direction
    model = load_model(MODEL, DEVICE)
    # queries depend only on subset (+prompt/val) -> build once
    q_false = prepare_data(PROMPT, VAL, subset='false')
    q_true = prepare_data(PROMPT, VAL, subset='true')
    print(f'queries: false={len(q_false)} true={len(q_true)}  hidden_states={len(hidden_states)} positions')

    nie_table = []
    for train_datasets, probe_name in CELLS:
        cell = '+'.join(train_datasets) + '/' + probe_name
        print(f'\n=== cell {cell} ===')
        direction, probe_cls_name = get_direction(train_datasets, probe_name)

        # PD-  : false subset, no intervention (probe-independent baseline)
        PDm, tot_PDm = intervention_experiment(model, q_false, direction, hidden_states, intervention='none')
        append_record(train_datasets, probe_cls_name, PDm, tot_PDm, 'none', 'false')
        # PD-* : false subset, add direction (false -> true)
        PDms, tot_PDms = intervention_experiment(model, q_false, direction, hidden_states, intervention='add')
        append_record(train_datasets, probe_cls_name, PDms, tot_PDms, 'add', 'false')
        # PD+  : true subset, no intervention
        PDp, tot_PDp = intervention_experiment(model, q_true, direction, hidden_states, intervention='none')
        append_record(train_datasets, probe_cls_name, PDp, tot_PDp, 'none', 'true')
        # PD+* : true subset, subtract direction (true -> false)
        PDps, tot_PDps = intervention_experiment(model, q_true, direction, hidden_states, intervention='subtract')
        append_record(train_datasets, probe_cls_name, PDps, tot_PDps, 'subtract', 'true')

        nie_ft = (PDms - PDm) / (PDp - PDm) if (PDp - PDm) != 0 else float('nan')
        nie_tf = (PDps - PDp) / (PDm - PDp) if (PDm - PDp) != 0 else float('nan')
        row = {
            'train_datasets': train_datasets, 'probe': probe_cls_name,
            'PD_minus': PDm, 'PD_minus_star': PDms, 'PD_plus': PDp, 'PD_plus_star': PDps,
            'NIE_false_to_true': nie_ft, 'NIE_true_to_false': nie_tf,
        }
        nie_table.append(row)
        print(f'  PD- {PDm:.4f}  PD-* {PDms:.4f}  PD+ {PDp:.4f}  PD+* {PDps:.4f}')
        print(f'  NIE false->true {nie_ft:.4f}   NIE true->false {nie_tf:.4f}')

    with open('experimental_outputs/nie_table_13b.json', 'w') as f:
        json.dump(nie_table, f, indent=2)
    print('\nwrote experimental_outputs/nie_table_13b.json')
    print('\n=== Table 2 (13B) NIE summary ===')
    print(f"{'cell':32s} {'NIE f->t':>9s} {'NIE t->f':>9s}")
    for r in nie_table:
        print(f"{'+'.join(r['train_datasets'])+'/'+r['probe']:32s} {r['NIE_false_to_true']:9.3f} {r['NIE_true_to_false']:9.3f}")


if __name__ == '__main__':
    main()
