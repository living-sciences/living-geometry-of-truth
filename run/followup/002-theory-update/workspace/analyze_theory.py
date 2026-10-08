"""002-theory-update: analysis-only study of how linear truth structure scales.

Reads 001-living-update artifacts (generalization_followup.json, flip_rates.json) plus
on-disk model configs. Fits accuracy-vs-scale, computes relative-depth of best layer,
negation gaps, generation/family effects, and the base->instruct tuning axis (incl. cosine
similarity of mass-mean directions computed from 001's activations on CPU).

No GPU, no re-extraction. Writes results/theory_update.json and three figures.
"""
import os, sys, json
import numpy as np
from scipy.optimize import curve_fit

FU001 = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/followup/001-living-update"
OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/followup/002-theory-update/results"
os.makedirs(OUT, exist_ok=True)

GEN = json.load(open(f"{FU001}/results/generalization_followup.json"))
FLIP = json.load(open(f"{FU001}/results/flip_rates.json"))

# ---- model metadata (params from safetensors total_size/2 on disk; n_layers from config.json) ----
# params in billions (bf16 bytes/2, includes untied lm_head where present)
META = {
    "llama-2-7b":          dict(params=6.738, n_layers=32, family="Llama", gen=2023, tuning="base"),
    "llama-2-13b":         dict(params=13.180, n_layers=40, family="Llama", gen=2023, tuning="base"),
    "llama-3.1-8b":        dict(params=8.030, n_layers=32, family="Llama", gen=2024, tuning="base"),
    "qwen2.5-7b":          dict(params=7.616, n_layers=28, family="Qwen",  gen=2024, tuning="base"),
    "qwen3-8b-base":       dict(params=8.192, n_layers=36, family="Qwen",  gen=2025, tuning="base"),
    "olmo-3-7b":           dict(params=7.298, n_layers=32, family="OLMo",  gen=2025, tuning="base"),
    "qwen2.5-7b-instruct": dict(params=7.616, n_layers=28, family="Qwen",  gen=2024, tuning="instruct"),
}

def at_best(m): return GEN[m]["at_best_layer"]

# collect per-model core numbers at best-val layer
rows = {}
for m, md in META.items():
    b = at_best(m)
    rows[m] = dict(
        params=md["params"], n_layers=md["n_layers"], family=md["family"],
        gen=md["gen"], tuning=md["tuning"],
        best_val_layer=GEN[m]["best_val_layer"], probe_layer_04=GEN[m]["probe_layer"],
        rel_depth=GEN[m]["best_val_layer"]/md["n_layers"],
        rel_depth_04=GEN[m]["probe_layer"]/md["n_layers"],
        C3_MM=b["C3_larger_smaller_to_sp_en_trans"]["MM"],
        C3_LR=b["C3_larger_smaller_to_sp_en_trans"]["LR"],
        C3_CCS=b["C3_larger_smaller_to_sp_en_trans"]["CCS"],
        C19_MM=b["C19_mean_offdiag"]["MM"],
        C19_LR=b["C19_mean_offdiag"]["LR"],
        oracle_MM=b["oracle_avg"]["MM"], oracle_LR=b["oracle_avg"]["LR"],
    )

print("=== per-model summary (best-val layer) ===")
hdr = "model            params  L  bestL  reld  rd04  C3_MM  C19_MM  orc_MM"
print(hdr)
for m,r in rows.items():
    print(f"{m:20s} {r['params']:5.2f} {r['n_layers']:2d} {r['best_val_layer']:4d}  {r['rel_depth']:.3f} {r['rel_depth_04']:.3f} {r['C3_MM']:.3f}  {r['C19_MM']:.3f}  {r['oracle_MM']:.3f}")

# ============ ANALYSIS 1: accuracy vs scale ============
# Base-only ladder for the scale fit (instruct handled on tuning axis).
base = [m for m in META if META[m]["tuning"]=="base"]
logP = np.array([np.log10(rows[m]["params"]) for m in base])
y19 = np.array([rows[m]["C19_MM"] for m in base])
y3 = np.array([rows[m]["C3_MM"] for m in base])

# log-linear fit: y = s*logP + b
sl19, in19 = np.polyfit(logP, y19, 1)
sl3, in3 = np.polyfit(logP, y3, 1)
# per-10x-params slope == slope per unit log10 == sl
r19 = np.corrcoef(logP, y19)[0,1]

# saturating exponential a - b*exp(-c*logP)
def sat(x,a,b,c): return a - b*np.exp(-c*x)
sat_fit = None
try:
    p0=[0.85, 0.5, 3.0]
    popt,_ = curve_fit(sat, logP, y19, p0=p0, maxfev=20000,
                       bounds=([0.5,0,0],[1.2,5,50]))
    yhat = sat(logP,*popt); ss_res=np.sum((y19-yhat)**2); ss_tot=np.sum((y19-y19.mean())**2)
    sat_fit = dict(a=float(popt[0]), b=float(popt[1]), c=float(popt[2]), r2=float(1-ss_res/ss_tot))
    # value at 8B and asymptote; saturation check: fraction of asymptote reached by 8B
    y8 = float(sat(np.log10(8.0),*popt))
    sat_frac_8B = y8/popt[0] if popt[0]>0 else float('nan')
    sat_fit.update(y_at_8B=y8, asymptote=float(popt[0]), sat_frac_8B=float(sat_frac_8B))
except Exception as e:
    sat_fit = dict(error=str(e))

# within-family clean scale signal: Llama-2 7B -> 13B (same generation)
llama2 = dict(
    C19_7b=rows["llama-2-7b"]["C19_MM"], C19_13b=rows["llama-2-13b"]["C19_MM"],
    C3_7b=rows["llama-2-7b"]["C3_MM"], C3_13b=rows["llama-2-13b"]["C3_MM"],
)

print("\n=== A1 accuracy vs scale (base ladder) ===")
print(f"C19 log-linear slope = {sl19:+.3f} per 10x params (r={r19:+.2f}); C3 slope = {sl3:+.3f}")
print(f"saturating exp fit: {sat_fit}")
print(f"Llama-2 within-family 7B->13B: C19 {llama2['C19_7b']:.3f}->{llama2['C19_13b']:.3f}, "
      f"C3 {llama2['C3_7b']:.3f}->{llama2['C3_13b']:.3f}")

# ============ ANALYSIS 2: best layer relative depth ============
rd_all = np.array([rows[m]["rel_depth"] for m in base])  # base models
depth = dict(
    mean=float(rd_all.mean()), sd=float(rd_all.std(ddof=1)),
    per_model={m: rows[m]["rel_depth"] for m in META},
    per_family={},
    rd04_mean=float(np.mean([rows[m]["rel_depth_04"] for m in base])),
)
for fam in ["Llama","Qwen","OLMo"]:
    vals=[rows[m]["rel_depth"] for m in base if META[m]["family"]==fam]
    depth["per_family"][fam]=dict(mean=float(np.mean(vals)), vals=vals)
print("\n=== A2 best-layer relative depth (base) ===")
print(f"mean {depth['mean']:.3f} +/- {depth['sd']:.3f}; per-family {[(f,round(depth['per_family'][f]['mean'],3)) for f in depth['per_family']]}")

# ============ ANALYSIS 3: negation gap (transformation generalization) ============
# MM: train on affirmative dataset, test on its negated counterpart, at best-val layer.
# gap = acc(aff->aff) - acc(aff->neg). Also reverse (neg->aff). Paper: negation flips truth dir.
def mm(m): return at_best(m)["matrix"]["MM"]
# Train rows are medleys: single-polarity probe evaluated on opposite polarity measures the
# negation flip (paper's C9/C12 geometry). We also record the polarity-agnostic medley recovery
# (cities+neg_cities trained -> neg_cities), the 2D-subspace signature of Buerger et al 2407.12831.
neg = {}
for m in META:
    M = mm(m)
    cc = M["cities"]["cities"]                 # affirmative probe, affirmative eval
    cn = M["cities"]["neg_cities"]             # affirmative probe, negated eval  -> flip
    cities_gap = cc - cn
    ls_aff = M["larger_than+smaller_than"]["sp_en_trans"]      # = C3
    ls_neg = M["larger_than+smaller_than"]["neg_sp_en_trans"]  # C3 probe on negated
    spen_gap = ls_aff - ls_neg
    both_to_neg = M["cities+neg_cities"]["neg_cities"]         # 2D-subspace recovery
    both_to_aff = M["cities+neg_cities"]["cities"]
    neg[m] = dict(
        cities_to_cities=cc, cities_to_neg=cn, cities_gap=cities_gap,
        c3_aff=ls_aff, c3_neg=ls_neg, spen_gap=spen_gap,
        mean_neg_gap=float(np.mean([cities_gap, spen_gap])),
        both_medley_to_neg=both_to_neg, both_medley_to_aff=both_to_aff,
    )
print("\n=== A3 negation gap (MM, best-val layer) ===")
print("model            cit->cit cit->neg cit_gap  c3_aff c3_neg spen_gap mean_gap  both->neg")
for m in META:
    n=neg[m]
    print(f"{m:20s} {n['cities_to_cities']:.3f}   {n['cities_to_neg']:.3f}   {n['cities_gap']:+.3f}  "
          f"{n['c3_aff']:.3f} {n['c3_neg']:.3f} {n['spen_gap']:+.3f}   {n['mean_neg_gap']:+.3f}   {n['both_medley_to_neg']:.3f}")

# ============ ANALYSIS 4: family / generation effects (base, ~7-8B) ============
def group_mean(pred, key):
    vals=[rows[m][key] for m in base if pred(m)]
    return float(np.mean(vals)), vals
by_gen={}
for gyr in [2023,2024,2025]:
    c19,_=group_mean(lambda m,gyr=gyr: META[m]["gen"]==gyr, "C19_MM")
    c3,_ =group_mean(lambda m,gyr=gyr: META[m]["gen"]==gyr, "C3_MM")
    by_gen[gyr]=dict(C19_MM=c19, C3_MM=c3, models=[m for m in base if META[m]["gen"]==gyr])
by_fam={}
for fam in ["Llama","Qwen","OLMo"]:
    c19,_=group_mean(lambda m,fam=fam: META[m]["family"]==fam, "C19_MM")
    by_fam[fam]=dict(C19_MM=c19, models=[m for m in base if META[m]["family"]==fam])
# 2025 vs 2024 at ~equal (7-8B) scale — exclude 13B (it's 2023 & bigger, already excluded)
print("\n=== A4 generation/family effects (base) ===")
print("by generation (C19 MM):", {g:round(by_gen[g]['C19_MM'],3) for g in by_gen})
print("by family (C19 MM):", {f:round(by_fam[f]['C19_MM'],3) for f in by_fam})

# ============ ANALYSIS 5: tuning axis (Qwen2.5-7B base vs instruct) ============
# accuracy deltas + flip-rate delta from JSON; cosine similarity of MM directions from acts.
sys.path.insert(0, f"{FU001}/workspace/codebase")
import torch as t
import pandas as pd
from glob import glob
from probes import MMProbe  # noqa (probes.py depends only on torch)
ACTS_ROOT = f"{FU001}/acts"
DATA = f"{FU001}/workspace/data/datasets_capped"
ACTS_BATCH_SIZE = 25

def collect_acts_dir(acts_root, model, dataset, layer, center=False, device="cpu"):
    # verbatim logic from 001 fu_common.collect_acts_dir (center default False here)
    directory = os.path.join(acts_root, model, dataset)
    files = glob(os.path.join(directory, f"layer_{layer}_*.pt"))
    n = ACTS_BATCH_SIZE * len(files)
    acts = [t.load(os.path.join(directory, f"layer_{layer}_{i}.pt")).to(device)
            for i in range(0, n, ACTS_BATCH_SIZE)]
    acts = t.cat(acts, dim=0).float().to(device)
    if center:
        acts = acts - t.mean(acts, dim=0)
    return acts

def load_labels(datasets_dir, dataset, device="cpu"):
    df = pd.read_csv(os.path.join(datasets_dir, f"{dataset}.csv"))
    return t.Tensor(df["label"].values).to(device)

def mm_direction(model, dataset, layer):
    acts = collect_acts_dir(ACTS_ROOT, model, dataset, layer, center=False, device="cpu")
    labs = load_labels(DATA, dataset, device="cpu")
    n = min(len(acts), len(labs)); acts, labs = acts[:n], labs[:n]
    p = MMProbe.from_data(acts, labs, device="cpu")
    d = p.direction.detach().float().numpy()
    return d/ (np.linalg.norm(d)+1e-9)

def cos(a,b): return float(np.dot(a,b))
tune = dict(
    C3_MM_base=rows["qwen2.5-7b"]["C3_MM"], C3_MM_instruct=rows["qwen2.5-7b-instruct"]["C3_MM"],
    C19_MM_base=rows["qwen2.5-7b"]["C19_MM"], C19_MM_instruct=rows["qwen2.5-7b-instruct"]["C19_MM"],
    oracle_MM_base=rows["qwen2.5-7b"]["oracle_MM"], oracle_MM_instruct=rows["qwen2.5-7b-instruct"]["oracle_MM"],
    flip_base=FLIP["qwen2.5-7b"]["flip_rate_overall_on_correct"],
    flip_instruct=FLIP["qwen2.5-7b-instruct"]["flip_rate_overall_on_correct"],
)
tune["C3_delta"]=tune["C3_MM_instruct"]-tune["C3_MM_base"]
tune["C19_delta"]=tune["C19_MM_instruct"]-tune["C19_MM_base"]
tune["flip_delta"]=tune["flip_instruct"]-tune["flip_base"]
# cosine sim at shared best-val layer (18) and at 0.4-depth layer (11) for cities and larger_than
cos_by={}
for L in [18, 11]:
    for ds in ["cities","larger_than","sp_en_trans"]:
        try:
            db=mm_direction("qwen2.5-7b", ds, L); di=mm_direction("qwen2.5-7b-instruct", ds, L)
            cos_by[f"L{L}_{ds}"]=cos(db,di)
        except Exception as e:
            cos_by[f"L{L}_{ds}"]=f"err:{e}"
tune["cos_MM_base_vs_instruct"]=cos_by
print("\n=== A5 tuning axis (Qwen2.5-7B base vs instruct) ===")
print(f"C3 {tune['C3_MM_base']:.3f}->{tune['C3_MM_instruct']:.3f} (d={tune['C3_delta']:+.3f}); "
      f"C19 {tune['C19_MM_base']:.3f}->{tune['C19_MM_instruct']:.3f} (d={tune['C19_delta']:+.3f}); "
      f"flip {tune['flip_base']:.3f}->{tune['flip_instruct']:.3f}")
print("cosine sim MM base vs instruct:", {k:(round(v,3) if isinstance(v,float) else v) for k,v in cos_by.items()})

# ============ save all ============
result = dict(
    per_model=rows,
    A1_scale=dict(C19_loglinear_slope_per10x=float(sl19), C19_intercept=float(in19), C19_corr=float(r19),
                  C3_loglinear_slope_per10x=float(sl3), C3_intercept=float(in3),
                  saturating_exp=sat_fit, llama2_within_family=llama2, base_models=base,
                  logP=logP.tolist(), C19=y19.tolist(), C3=y3.tolist()),
    A2_depth=depth,
    A3_negation=neg,
    A4_gen_family=dict(by_generation=by_gen, by_family=by_fam),
    A5_tuning=tune,
)
json.dump(result, open(f"{OUT}/theory_update.json","w"), indent=2)
print(f"\nwrote {OUT}/theory_update.json")
