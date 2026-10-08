"""Faithful port of generalization.ipynb result cells (In[31] LR/MM medleys, CCS cell,
In[25] oracle) plus per-model heatmap rendering (In[16]), run for BOTH llama-2-13b and
llama-2-7b. Appends per-probe accuracy dicts to experimental_outputs/generalization_results.json
and saves figures/<model>_generalization.pdf.

70B is omitted (no local weights); the notebook's 3-model scale-trend line plots (In[8]/In[6])
are not rendered here because they require the 70B column. The 7B-vs-13B trend is available
from the appended JSON and from the printed OOD averages below.
"""
import torch as t
import random
import json
import configparser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import DataManager
from probes import LRProbe, MMProbe, CCSProbe

device = 'cuda:0' if t.cuda.is_available() else 'cpu'
split = 0.8
SEED = 42  # fixed for reproducibility (train/val split seed is a setup value, not a result)

config = configparser.ConfigParser()
config.read('config.ini')

train_medlies = [
    ['cities'],
    ['cities', 'neg_cities'],
    ['larger_than'],
    ['larger_than', 'smaller_than'],
    ['likely'],
]
val_datasets = [
    'cities', 'neg_cities', 'larger_than', 'smaller_than', 'sp_en_trans',
    'neg_sp_en_trans', 'cities_cities_conj', 'cities_cities_disj',
    'companies_true_false', 'common_claim_true_false', 'counterfact_true_false',
]
ccs_medlies = [['cities', 'neg_cities'], ['larger_than', 'smaller_than']]
ProbeClasses = [LRProbe, MMProbe]

def to_str(l):
    return '+'.join(l)

RESULTS = 'experimental_outputs/generalization_results.json'

def append_out(out):
    with open(RESULTS, 'r') as f:
        outs = json.load(f)
    outs.append(out)
    with open(RESULTS, 'w') as f:
        json.dump(outs, f, indent=2)


def run_model(model):
    layer = eval(config[model]['probe_layer'])
    noperiod = eval(config[model]['noperiod'])
    seed = SEED
    print(f'\n===== {model} (layer {layer}, noperiod {noperiod}, seed {seed}) =====')

    # ---- LR / MM medley generalization (In[31]) ----
    accs = {pc.__str__(): {to_str(m): {} for m in train_medlies} for pc in ProbeClasses}
    for ProbeClass in ProbeClasses:
        for medley in train_medlies:
            dm = DataManager()
            for dataset in medley:
                dm.add_dataset(dataset, model, layer, split=split, seed=seed, noperiod=noperiod, center=True, device=device)
            for dataset in val_datasets:
                if dataset not in medley:
                    dm.add_dataset(dataset, model, layer, split=None, noperiod=noperiod, center=True, device=device)
            train_acts, train_labels = dm.get('train')
            probe = ProbeClass.from_data(train_acts, train_labels, device=device)
            for val_dataset in val_datasets:
                if val_dataset in medley:
                    a, lb = dm.data['val'][val_dataset]
                    accs[ProbeClass.__str__()][to_str(medley)][val_dataset] = (probe.pred(a, iid=True) == lb).float().mean().item()
                else:
                    a, lb = dm.data[val_dataset]
                    accs[ProbeClass.__str__()][to_str(medley)][val_dataset] = (probe.pred(a, iid=False) == lb).float().mean().item()
    for ProbeClass in ProbeClasses:
        out = dict(accs[ProbeClass.__str__()])
        out['model'] = model; out['probe'] = ProbeClass.__str__(); out['layer'] = layer
        out['oracle'] = False; out['noperiod'] = noperiod
        append_out(out)

    # ---- CCS paired medleys ----
    ccs_accs = {to_str(m): {} for m in ccs_medlies}
    for medley in ccs_medlies:
        dm = DataManager()
        for dataset in medley:
            dm.add_dataset(dataset, model, layer, split=split, seed=seed, noperiod=noperiod, center=True, device=device)
        for dataset in val_datasets:
            if dataset not in medley:
                dm.add_dataset(dataset, model, layer, split=None, noperiod=noperiod, center=True, device=device)
        train_acts, train_labels = dm.data['train'][medley[0]]
        train_neg_acts, _ = dm.data['train'][medley[1]]
        probe = CCSProbe.from_data(train_acts, train_neg_acts, train_labels, device=device)
        for val_dataset in val_datasets:
            if val_dataset in medley:
                a, lb = dm.data['val'][val_dataset]
            else:
                a, lb = dm.data[val_dataset]
            ccs_accs[to_str(medley)][val_dataset] = (probe.pred(a) == lb).float().mean().item()
    out = dict(ccs_accs)
    out['model'] = model; out['probe'] = 'CCSProbe'; out['layer'] = layer
    out['oracle'] = False; out['noperiod'] = noperiod
    append_out(out)

    # ---- oracle (train/test same dataset) ----
    oracle_accs = {pc.__str__(): {} for pc in ProbeClasses}
    for ProbeClass in ProbeClasses:
        for dataset in val_datasets:
            dm = DataManager()
            dm.add_dataset(dataset, model, layer, split=split, noperiod=noperiod, seed=seed, device=device)
            a, lb = dm.get('train')
            probe = ProbeClass.from_data(a, lb, device=device)
            a, lb = dm.data['val'][dataset]
            oracle_accs[ProbeClass.__str__()][dataset] = (probe(a, iid=True).round() == lb).float().mean().item()
    for ProbeClass in ProbeClasses:
        out = dict(oracle_accs[ProbeClass.__str__()])
        out['model'] = model; out['probe'] = ProbeClass.__str__(); out['oracle'] = True
        out['layer'] = layer; out['noperiod'] = noperiod
        append_out(out)

    # ---- OOD averages (7B-vs-13B trend scalars) ----
    def avg(d, drop=()):
        vals = [v for k, v in d.items() if k not in drop]
        return sum(vals) / len(vals)
    print('  LR  cities            OOD-avg:', round(avg(accs['LRProbe']['cities'], drop=('cities',)), 4))
    print('  LR  cities+neg_cities OOD-avg:', round(avg(accs['LRProbe']['cities+neg_cities'], drop=('cities','neg_cities')), 4))
    print('  LR  larger_than       OOD-avg:', round(avg(accs['LRProbe']['larger_than'], drop=('larger_than',)), 4))
    print('  LR  lt+st             OOD-avg:', round(avg(accs['LRProbe']['larger_than+smaller_than'], drop=('larger_than','smaller_than')), 4))
    print('  MM  cities+neg_cities OOD-avg:', round(avg(accs['MMProbe']['cities+neg_cities'], drop=('cities','neg_cities')), 4))
    print('  CCS cities+neg_cities OOD-avg:', round(avg(ccs_accs['cities+neg_cities'], drop=('cities','neg_cities')), 4))
    print('  oracle LR avg        :', round(avg(oracle_accs['LRProbe']), 4))

    # ---- heatmap grid (In[16]) ----
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    grids = {
        0: ('LRProbe', accs['LRProbe'], [['cities'], ['cities','neg_cities'], ['larger_than'], ['larger_than','smaller_than']]),
        1: ('MMProbe', accs['MMProbe'], [['cities'], ['cities','neg_cities'], ['larger_than'], ['larger_than','smaller_than']]),
        2: ('CCSProbe', ccs_accs, [['cities','neg_cities'], ['larger_than','smaller_than']]),
    }
    for axi, (title, src, medlies) in grids.items():
        ax = axes[axi]
        grid = [[src[to_str(m)][d] for m in medlies] for d in val_datasets]
        ax.imshow(grid, vmin=0, vmax=1, aspect='auto')
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                ax.text(j, i, f'{round(grid[i][j]*100):2d}', ha='center', va='center', fontsize=7)
        ax.set_xticks(range(len(medlies)))
        ax.set_xticklabels([to_str(m) for m in medlies], rotation=90, fontsize=6)
        ax.set_title(title, fontsize=9)
        if axi == 0:
            ax.set_yticks(range(len(val_datasets)))
            ax.set_yticklabels(val_datasets, fontsize=6)
        else:
            ax.set_yticks([])
    fig.suptitle(f'{model} generalization (val-dataset x train-set x technique)')
    fig.tight_layout()
    fig.savefig(f'figures/{model}_generalization.pdf', bbox_inches='tight')
    fig.savefig(f'figures/{model}_generalization.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  saved figures/{model}_generalization.pdf/.png')


if __name__ == '__main__':
    t.set_grad_enabled(True)  # probes need grad
    for model in ['llama-2-13b', 'llama-2-7b']:
        run_model(model)
    print('\nDONE')
