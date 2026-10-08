"""Build the 003 followup figures (extends 001 to the 2026 Qwen3.5-9B-Base point).
  results/truth_direction_over_time.png  (C3 gen acc vs release era; anchor line at 0.9746; 2026 pt)
  results/probe_acc_vs_scale.png         (mean OOD acc vs log params, colored by family)
  results/pca_cities_grid.png            (top-2 PC cities scatter per model incl 2026; C1)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch as t
from fu_common import collect_acts_dir
from utils import get_pcs

FU = '/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/followup/003-living-update'
REPL = '/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/replication/codebase'

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

# (release_decimal_year, params_B, family, label, is_instruct, is_2026)
META = {
    'llama-2-7b':          (2023.5, 7.0,  'Llama',  'Llama-2-7B', False, False),
    'llama-2-13b':         (2023.5, 13.0, 'Llama',  'Llama-2-13B', False, False),
    'llama-3.1-8b':        (2024.6, 8.0,  'Llama',  'Llama-3.1-8B', False, False),
    'qwen2.5-7b':          (2024.7, 7.6,  'Qwen',   'Qwen2.5-7B', False, False),
    'qwen3-8b-base':       (2025.3, 8.2,  'Qwen',   'Qwen3-8B-Base', False, False),
    'olmo-3-7b':           (2025.8, 7.3,  'OLMo',   'OLMo-3-7B', False, False),
    'qwen2.5-7b-instruct': (2024.7, 7.6,  'Qwen',   'Qwen2.5-7B-Instruct', True, False),
    'qwen3.5-9b-base':     (2026.2, 9.0,  'Qwen',   'Qwen3.5-9B-Base', False, True),  # Mar 2026
    'gemma-4-12b':         (2026.4, 12.0, 'Gemma',  'Gemma-4-12B', False, True),      # 2026
}
FAM_COLOR = {'Llama': OKABE_ITO[0], 'Qwen': OKABE_ITO[1], 'OLMo': OKABE_ITO[2], 'Gemma': OKABE_ITO[4]}
EMPH = OKABE_ITO[3]  # orange-red emphasis for the new 2026 model

res = json.load(open(f'{FU}/results/generalization_followup.json'))
ANCHOR_MM = 0.9746  # llama-2-13b replicated C3 (MM), from replication generalization_results.json (disk)


def c3(model, which, layer='at_best_layer'):
    return res[model][layer]['C3_larger_smaller_to_sp_en_trans'][which]


# ---------- Figure 1: truth direction over time ----------
base = [m for m in META if not META[m][4]]
base = sorted(base, key=lambda m: META[m][0])
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.axhline(ANCHOR_MM, color='gray', ls='--', lw=1.2, zorder=0,
           label=f'Llama-2-13B paper anchor (MM {ANCHOR_MM}, replication on disk)')
for which, color, mk, lab in [('MM', OKABE_ITO[3], 'o', 'MM (mass-mean)'),
                              ('LR', OKABE_ITO[0], 's', 'LR'),
                              ('CCS', OKABE_ITO[2], '^', 'CCS')]:
    xs = [META[m][0] for m in base]
    ys = [c3(m, which) for m in base]
    ax.plot(xs, ys, mk, color=color, ms=8, ls='-' if which == 'MM' else 'None',
            lw=1.5, label=lab, zorder=3)
# emphasize the 2026 points (MM), plot at both best-val (filled) and 0.4-depth (x)
new26 = [m for m in base if META[m][5]]
ax.scatter([META[m][0] for m in new26], [c3(m, 'MM') for m in new26],
           s=230, facecolor='none', edgecolor='black', lw=2.2, zorder=5,
           label='2026 models (new; MM @ best-val layer)')
ax.scatter([META[m][0] for m in new26], [c3(m, 'MM', 'at_probe_layer') for m in new26],
           s=70, marker='x', color='black', lw=1.6, zorder=6,
           label="2026 MM @ paper 0.4-depth layer")
# base<->instruct connected pair (MM)
qb, qi = 'qwen2.5-7b', 'qwen2.5-7b-instruct'
ax.plot([META[qb][0], META[qi][0] + 0.12], [c3(qb, 'MM'), c3(qi, 'MM')],
        color=OKABE_ITO[4], ls=':', lw=1.5, zorder=2)
ax.plot(META[qi][0] + 0.12, c3(qi, 'MM'), 'D', color=OKABE_ITO[4], ms=8,
        label='Qwen2.5-7B-Instruct (chat)', zorder=4)
for m in base:
    dy = 12 if META[m][5] else 8
    ax.annotate(META[m][3], (META[m][0], c3(m, 'MM')), fontsize=7.5,
                xytext=(0, dy), textcoords='offset points', ha='center')
ax.annotate('Instruct', (META[qi][0] + 0.12, c3(qi, 'MM')), fontsize=7.5,
            xytext=(6, -12), textcoords='offset points')
ax.set_xlabel('Model release (year)')
ax.set_ylabel('C3 cross-dataset generalization accuracy\n(larger+smaller → sp_en_trans)')
ax.set_ylim(0.33, 1.04)
ax.set_xlim(2023.2, 2026.8)
ax.legend(fontsize=8.5, loc='lower right', framealpha=0.95)
fig.savefig(f'{FU}/results/truth_direction_over_time.png', facecolor='white')
plt.close(fig)
print('wrote truth_direction_over_time.png')

# ---------- Figure 2: probe acc vs scale ----------
fig, ax = plt.subplots(figsize=(8, 5))
for m in META:
    yr, p, fam, lab, inst, is26 = META[m]
    y = res[m]['at_best_layer']['C19_mean_offdiag']['MM']
    ax.scatter(p, y, s=150 if is26 else 110, color=FAM_COLOR[fam],
               marker='*' if is26 else ('D' if inst else 'o'),
               edgecolor=EMPH if is26 else 'black', lw=1.6 if is26 else 0.6, zorder=3)
    ax.annotate(lab, (p, y), fontsize=7.5, xytext=(0, 9 if is26 else 8),
                textcoords='offset points', ha='center')
ax.set_xscale('log')
ax.set_xlabel('Parameters (billions, log scale)')
ax.set_ylabel('Mean OOD generalization accuracy\n(MM off-diagonal, C19)')
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=FAM_COLOR[f],
                  markeredgecolor='black', ms=10, label=f) for f in FAM_COLOR]
handles.append(Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                      markeredgecolor='black', ms=10, label='instruct-tuned'))
handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor=FAM_COLOR['Qwen'],
                      markeredgecolor=EMPH, ms=14, label='2026 model (new)'))
ax.legend(handles=handles, fontsize=9, loc='lower right')
fig.savefig(f'{FU}/results/probe_acc_vs_scale.png', facecolor='white')
plt.close(fig)
print('wrote probe_acc_vs_scale.png')

# ---------- Figure 3: PCA cities grid ----------
models = ['llama-2-13b', 'llama-3.1-8b', 'qwen3-8b-base', 'olmo-3-7b', 'qwen3.5-9b-base', 'gemma-4-12b']
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
for ax, m in zip(axes.flat, models):
    layer = res[m]['best_val_layer']
    acts_root = f'{REPL}/acts' if m.startswith('llama-2') else f'{FU}/acts'
    ds_dir = f'{REPL}/datasets' if m.startswith('llama-2') else f'{FU}/workspace/data/datasets_capped'
    acts = collect_acts_dir(acts_root, m, 'cities', layer, center=True)
    labels = pd.read_csv(f'{ds_dir}/cities.csv')['label'].values
    pcs = get_pcs(acts, k=2)
    proj = (acts @ pcs).numpy()
    for lb, c, name in [(1, OKABE_ITO[0], 'true'), (0, OKABE_ITO[3], 'false')]:
        mask = labels == lb
        ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.5, color=c, label=name)
    ttl = f'{META[m][3]} (L{layer})'
    ax.set_title(ttl, fontsize=10, color=EMPH if META[m][5] else 'black',
                 fontweight='bold' if META[m][5] else 'normal')
    ax.set_xlabel('PC1', fontsize=9); ax.set_ylabel('PC2', fontsize=9)
    ax.tick_params(labelsize=7)
axes.flat[0].legend(fontsize=8, markerscale=2)
fig.suptitle('Top-2 PCA of cities activations, colored by truth label (C1) — 2026 model in orange', fontsize=13)
fig.tight_layout()
fig.savefig(f'{FU}/results/pca_cities_grid.png', facecolor='white')
plt.close(fig)
print('wrote pca_cities_grid.png')
