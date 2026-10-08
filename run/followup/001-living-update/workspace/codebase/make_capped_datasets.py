"""Create N<=1000 capped datasets for new-model extraction (seed 0).
Paired datasets (cities/neg_cities, larger_than/smaller_than) share sampled row indices
so CCS pairing and negation alignment are preserved. sp_en_trans/neg_sp_en_trans use all
rows (354). Writes to workspace/data/datasets_capped/<name>.csv preserving all columns.
"""
import os, numpy as np, pandas as pd

SRC = 'datasets'  # symlink -> replication full datasets
OUT = '../data/datasets_capped'
CAP = 1000
SEED = 0
os.makedirs(OUT, exist_ok=True)

def sample_idx(n, cap, seed):
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=cap, replace=False)
    idx.sort()  # keep ascending order (batching/label order stable)
    return idx

# paired groups share indices
pairs = [('cities', 'neg_cities'), ('larger_than', 'smaller_than')]
singles = ['likely', 'cities_cities_conj', 'sp_en_trans', 'neg_sp_en_trans']

def save(name, idx):
    df = pd.read_csv(f'{SRC}/{name}.csv')
    df.iloc[idx].reset_index(drop=True).to_csv(f'{OUT}/{name}.csv', index=False)
    print(f'{name}: {len(df)} -> {len(idx)}')

for a, b in pairs:
    na = len(pd.read_csv(f'{SRC}/{a}.csv'))
    idx = sample_idx(na, CAP, SEED)
    save(a, idx); save(b, idx)

for name in singles:
    n = len(pd.read_csv(f'{SRC}/{name}.csv'))
    idx = sample_idx(n, CAP, SEED)
    save(name, idx)

print('done')
