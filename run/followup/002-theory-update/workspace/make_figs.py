"""Figures for 002-theory-update. Reads results/theory_update.json (produced by analyze_theory.py)."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/geometry-of-truth/run/followup/002-theory-update/results"
R = json.load(open(f"{OUT}/theory_update.json"))
PM = R["per_model"]

FAMC = {"Llama": "#4C72B0", "Qwen": "#DD8452", "OLMo": "#55A868"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})

LABEL = {"llama-2-7b":"Llama-2-7B\n(2023)","llama-2-13b":"Llama-2-13B\n(2023, anchor)",
         "llama-3.1-8b":"Llama-3.1-8B\n(2024)","qwen2.5-7b":"Qwen2.5-7B\n(2024)",
         "qwen3-8b-base":"Qwen3-8B-Base\n(2025)","olmo-3-7b":"OLMo-3-7B\n(2025)",
         "qwen2.5-7b-instruct":"Qwen2.5-7B-Instruct"}

# ---------- Figure A: accuracy (C19 OOD MM) vs scale w/ fit ----------
A1 = R["A1_scale"]
base = A1["base_models"]
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for m in base:
    fam = PM[m]["family"]
    ax.scatter(PM[m]["params"], PM[m]["C19_MM"], s=140, color=FAMC[fam],
               edgecolor="black", linewidth=0.8, zorder=5)
    # nudge labels to avoid the two overlapping ~8B points
    off = {"llama-2-7b":(0,9,"center"), "llama-2-13b":(0,9,"center"),
           "llama-3.1-8b":(-24,6,"right"), "qwen3-8b-base":(26,4,"left"),
           "qwen2.5-7b":(-6,-20,"center"), "olmo-3-7b":(0,-20,"center")}[m]
    ax.annotate(m.replace("-base",""), (PM[m]["params"], PM[m]["C19_MM"]),
                textcoords="offset points", xytext=(off[0], off[1]),
                ha=off[2], fontsize=8)
# instruct point (open marker, on tuning axis, not in fit)
mi="qwen2.5-7b-instruct"
ax.scatter(PM[mi]["params"], PM[mi]["C19_MM"], s=150, facecolor="white",
           edgecolor=FAMC["Qwen"], linewidth=2, marker="D", zorder=5)
ax.annotate("Qwen2.5-7B-Instruct", (PM[mi]["params"], PM[mi]["C19_MM"]),
            textcoords="offset points", xytext=(0,10), ha="center", fontsize=8, color=FAMC["Qwen"])
# fits over param range
xg = np.linspace(6.3, 13.6, 100); lxg = np.log10(xg)
ax.plot(xg, A1["C19_loglinear_slope_per10x"]*lxg + A1["C19_intercept"], "--", color="0.35",
        label=f"log-linear: +{A1['C19_loglinear_slope_per10x']:.2f}/10x params (r={A1['C19_corr']:+.2f})")
sf = A1["saturating_exp"]
ax.plot(xg, sf["a"] - sf["b"]*np.exp(-sf["c"]*lxg), "-", color="#8172B3", alpha=0.8,
        label=f"saturating: a-b·e^(-c·logP), a={sf['a']:.2f}, R²={sf['r2']:.2f}")
ax.axhline(PM["llama-2-13b"]["C19_MM"], color="0.6", lw=1, ls=":")
ax.text(6.35, PM["llama-2-13b"]["C19_MM"]+0.006, "Llama-2-13B anchor (0.769)", fontsize=8, color="0.4")
# within-family Llama-2 scale arrow
ax.annotate("", xy=(PM["llama-2-13b"]["params"], PM["llama-2-13b"]["C19_MM"]),
            xytext=(PM["llama-2-7b"]["params"], PM["llama-2-7b"]["C19_MM"]),
            arrowprops=dict(arrowstyle="->", color=FAMC["Llama"], lw=1.4, alpha=0.6))
ax.set_xscale("log"); ax.set_xticks([7,8,9,10,13]); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("Parameters (B, log scale)"); ax.set_ylabel("Mean OOD generalization (MM off-diagonal, C19)")
ax.set_title("Truth-direction generalization vs scale\n(within-family Llama-2 7B→13B gains dominate; 7–8B moderns plateau ~0.8)")
handles=[Line2D([],[],marker='o',ls='',color=c,mec='k',label=f) for f,c in FAMC.items()]
handles.append(Line2D([],[],marker='D',ls='',mfc='white',mec=FAMC['Qwen'],mew=2,label='instruct (tuning axis)'))
leg1=ax.legend(handles=handles, loc="lower right", fontsize=8, title="family")
ax.add_artist(leg1); ax.legend(loc="lower left", fontsize=8)
ax.set_ylim(0.45, 0.90)
fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_vs_scale.png"); plt.close(fig)

# ---------- Figure B: best-layer relative depth per model / family ----------
D = R["A2_depth"]
order = ["llama-2-7b","llama-2-13b","llama-3.1-8b","qwen2.5-7b","qwen3-8b-base","olmo-3-7b","qwen2.5-7b-instruct"]
fig, ax = plt.subplots(figsize=(8.2, 5.0))
xs=np.arange(len(order))
for i,m in enumerate(order):
    fam=PM[m]["family"]
    hatch = "//" if PM[m]["tuning"]=="instruct" else None
    ax.bar(i, PM[m]["rel_depth"], color=FAMC[fam], edgecolor="black", hatch=hatch, zorder=3)
    ax.text(i, PM[m]["rel_depth"]+0.012, f"{PM[m]['rel_depth']:.2f}\n(L{PM[m]['best_val_layer']}/{PM[m]['n_layers']})",
            ha="center", fontsize=8)
# paper Llama-2 band 0.35-0.41
ax.axhspan(0.35, 0.41, color=FAMC["Llama"], alpha=0.12, zorder=0)
ax.text(4.0, 0.31, "paper Llama-2 band 0.35–0.41", ha="center", va="center", fontsize=8, color=FAMC["Llama"])
ax.axhline(0.4, color="0.4", ls="--", lw=1, zorder=1, label="fixed 0.4-depth cut (001)")
ax.set_xticks(xs); ax.set_xticklabels([LABEL[m].replace("\n"," ") for m in order], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Best-validation probe layer / model depth")
ax.set_title(f"Truth is read out at family-specific relative depth\nLlama ~{D['per_family']['Llama']['mean']:.2f} · Qwen ~{D['per_family']['Qwen']['mean']:.2f} · OLMo ~{D['per_family']['OLMo']['mean']:.2f}  (base mean {D['mean']:.2f}±{D['sd']:.2f})")
ax.set_ylim(0,0.8); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/best_relative_depth.png"); plt.close(fig)

# ---------- Figure C: negation gap vs generation ----------
N = R["A3_negation"]
fig, ax = plt.subplots(figsize=(7.8, 5.2))
jitter={2023:-0.12,2024:0.0,2025:0.12}
seen=set()
for m in order:
    if PM[m]["tuning"]=="instruct":
        continue
    g=PM[m]["gen"]; fam=PM[m]["family"]
    x=g + (0.28 if m=="llama-2-13b" else jitter[g]*0)  # spread within a year
for m in [x for x in order if PM[x]["tuning"]=="base"]:
    g=PM[m]["gen"]; fam=PM[m]["family"]
    # small x offset by family so points don't overlap within a year
    off={"Llama":-0.18,"Qwen":0.0,"OLMo":0.18}[fam]
    if m=="llama-2-13b": off=0.18
    lab = fam if fam not in seen else None; seen.add(fam)
    ax.scatter(g+off, N[m]["cities_gap"], s=150, color=FAMC[fam], edgecolor="black", zorder=5, label=lab)
    ax.annotate(m.replace("-base",""), (g+off, N[m]["cities_gap"]), textcoords="offset points",
                xytext=(0,10), ha="center", fontsize=7.5)
# instruct as diamond at 2024
ax.scatter(2024+0.30, N["qwen2.5-7b-instruct"]["cities_gap"], s=150, marker="D",
           facecolor="white", edgecolor=FAMC["Qwen"], linewidth=2, zorder=5)
ax.annotate("Qwen2.5-Instruct",(2024+0.30,N["qwen2.5-7b-instruct"]["cities_gap"]),
            textcoords="offset points", xytext=(0,10), ha="center", fontsize=7.5, color=FAMC["Qwen"])
ax.axhline(0, color="0.6", lw=1)
ax.set_xticks([2023,2024,2025]); ax.set_xlabel("Model release generation")
ax.set_ylabel("Negation gap  (MM cities→cities  −  cities→neg_cities)")
ax.set_title("Negation geometry aligns in newer models\naffirmative truth direction increasingly survives negation (2-D subspace, 2407.12831)", fontsize=10.5)
ax.legend(title="family", fontsize=8, loc="upper right")
ax.set_ylim(-0.03, 0.6)
fig.tight_layout(); fig.savefig(f"{OUT}/negation_gap_vs_generation.png"); plt.close(fig)

print("wrote 3 figures to", OUT)
