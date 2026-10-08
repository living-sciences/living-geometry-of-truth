"""Patch dataexplorer.ipynb for llama-2-13b, fix the outdated model_size kwarg,
and redirect every figure to figures/. Writes dataexplorer_13b.ipynb."""
import nbformat, os

nb = nbformat.read('dataexplorer.ipynb', as_version=4)
os.makedirs('figures', exist_ok=True)

def sub(src, a, b):
    assert a in src, f"pattern not found: {a!r}"
    return src.replace(a, b)

cells = [c for c in nb.cells if c.cell_type == 'code']

# Cell 0: config / model selection
cells[0].source = sub(cells[0].source, "model = 'llama-2-70b'", "model = 'llama-2-13b'")
cells[0].source += "\nimport os\nos.makedirs('figures', exist_ok=True)\nprint('config layer', layer, 'noperiod', noperiod, 'device', device)"

# Cell 1: cities + neg_cities in top-2 PCs -> assign to fig and save
cells[1].source = sub(cells[1].source, "model='llama-2-70b'", "model='llama-2-13b'")
cells[1].source = "fig = " + cells[1].source.lstrip()
cells[1].source += "\nfig.write_image('figures/pca_cities_neg_cities.png', scale=2)\nprint('saved pca_cities_neg_cities.png')"

# Cell 2: all 11 datasets, each own PCA basis. Last expr is fig.update_layout(...). Save fig.
cells[2].source += "\nfig.write_image('figures/all_datasets_pca.png', scale=2)\nprint('saved all_datasets_pca.png')"

# Cell 3: cross-basis projection. Ends with fig.show(). Save before show.
cells[3].source = sub(cells[3].source, "fig.show()",
                      "fig.write_image('figures/cross_basis.png', scale=2)\nprint('saved cross_basis.png')\nfig.show()")

# Cell 4: negation pairs at layer 12. Fix model_size -> model; save.
cells[4].source = sub(cells[4].source, "model_size='13B'", "model='llama-2-13b'")
# Fix pandas-version regression: `td.df.loc[pair[0], 'label'] = ...apply(...)` misaligns
# on the MultiIndex under modern pandas and sets the ENTIRE label column to NaN (all points
# one color + kaleido null-crash). Replace with a positional numpy remap preserving intent.
cells[4].source = sub(
    cells[4].source,
    "    td.df.loc[pair[0], 'label'] = td.df.loc[pair[0]]['label'].apply(lambda x: BLUE if x == 1 else RED)",
    "    import numpy as _np\n"
    "    _lab = td.df['label'].values\n"
    "    _m0 = td.df.index.get_level_values(0) == pair[0]\n"
    "    td.df['label'] = _np.where(_m0, _np.where(_lab == 1, BLUE, RED), _np.where(_lab == 1, YELLOW, PURPLE))")
cells[4].source = sub(
    cells[4].source,
    "    td.df.loc[pair[1], 'label'] = td.df.loc[pair[1]]['label'].apply(lambda x: YELLOW if x == 1 else PURPLE)",
    "")
cells[4].source = sub(cells[4].source, "fig.show()",
                      "fig.write_image('figures/negation_pairs_layer12.png', scale=2)\nprint('saved negation_pairs_layer12.png')\nfig.show()")

# Cell 5: layer sweep. Fix model_size -> model; redirect junk/ -> figures/.
cells[5].source = sub(cells[5].source, "model_size='13B'", "model='llama-2-13b'")
cells[5].source = sub(cells[5].source, "fig.write_image('junk/layer_sweep.png')",
                      "fig.write_image('figures/layer_sweep.png', scale=2)\nprint('saved layer_sweep.png')")

# New cell 6: rotation of the cities/neg_cities truth axis across layers (early/mid/late).
rotation_cell = nbformat.v4.new_code_cell(source='''
# Axis-rotation panel: cities + neg_cities in their own PCA basis at early/mid/late layers.
from plotly.subplots import make_subplots
from visualization_utils import TruthData
RED, BLUE, PURPLE, YELLOW = 1, .17, 0, .73
rot_layers = [4, 12, 24]
rfig = make_subplots(rows=1, cols=len(rot_layers), shared_yaxes=True,
                     x_title='PC1', y_title='PC2',
                     subplot_titles=[f'layer {L}' for L in rot_layers])
for j, L in enumerate(rot_layers):
    import numpy as _np
    td = TruthData.from_datasets(['cities', 'neg_cities'], model='llama-2-13b', layer=L, device=device)
    _lab = td.df['label'].values
    _m0 = td.df.index.get_level_values(0) == 'cities'
    td.df['label'] = _np.where(_m0, _np.where(_lab == 1, BLUE, RED), _np.where(_lab == 1, YELLOW, PURPLE))
    sub = td.plot(dimensions=2, color='label')
    sub.data[0]['showlegend'] = False
    rfig.add_trace(sub.data[0], row=1, col=j+1)
    # also save an individual panel per layer
    sfig = td.plot(dimensions=2, color='label')
    sfig.update_coloraxes(colorscale='Rainbow')
    sfig.write_image(f'figures/rotation_layer_{L}.png', scale=2)
rfig.update_coloraxes(colorscale='Rainbow')
rfig.update_layout(height=450, width=1200, coloraxis_showscale=False,
                   title={'text': 'cities+neg_cities truth-axis rotation across layers', 'font': {'size': 20}})
rfig.write_image('figures/rotation_layers_panel.png', scale=2)
print('saved rotation_layer_{4,12,24}.png and rotation_layers_panel.png')
''')
nb.cells.append(rotation_cell)

nbformat.write(nb, 'dataexplorer_13b.ipynb')
print('wrote dataexplorer_13b.ipynb with', len(nb.cells), 'cells')
