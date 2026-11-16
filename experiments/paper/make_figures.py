"""Figures for the manuscript, drawn only from results/*.csv / *.json."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results"); F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../paper/figures")
os.makedirs(F, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]; INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "legend.frameon": False, "figure.dpi": 200})

# Fig 1: R5b, Brier vs duplication level (h = 0.5)
d = pd.read_csv(f"{R}/r5b/r5b_cells.csv"); x = d[d.h == .5]
meths = [("stacking", "count stacking (D=1)", "o"), ("presence", "presence", "s"), ("a4", "A4 (midpoint)", "^"),
         ("augmented", "augmented stacking", "D"), ("matched", "matched (oracle)", "v")]
fig, ax = plt.subplots(figsize=(3.4, 3.0))
Ds = sorted(x.D.unique())
for (m, lab, mk), c in zip(meths, C):
    y = [x[x.D == D][f"{m}_brier"].mean() for D in Ds]
    ax.plot(range(len(Ds)), y, color=c, lw=2, marker=mk, ms=4.5, label=lab)
ax.set_xticks(range(len(Ds)), [str(D) for D in Ds]); ax.set_xlabel("copies per firing alarm, D (test)"); ax.set_ylabel("Brier score")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.45, -.28), ncol=2); fig.tight_layout(); fig.savefig(f"{F}/fig_r5b_duplication.pdf"); plt.close(fig)

# Fig 2: R6 primary effect per held-out block
j = json.load(open(f"{R}/r6/r6_confirmatory.json")); p1 = j["P1_la_aug_vs_augmented_unseen"]
pb = {int(k): v for k, v in p1["per_block"].items()}; ks = sorted(pb)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.bar(range(len(ks)), [pb[k] for k in ks], color=C[0], width=.6)
ax.axhline(p1["est"], color=INK, lw=1); ax.axhspan(*p1["t_ci95"], color=C[0], alpha=.12, lw=0)
ax.axhline(0, color=INK2, lw=.8)
ax.set_title(f"mean {p1['est']:.4f} (line), 95% t-interval [{p1['t_ci95'][0]:.4f}, {p1['t_ci95'][1]:.4f}] (band)", fontsize=7, color=INK)
ax.set_xticks(range(len(ks)), [f"block {k}" for k in ks], fontsize=7); ax.set_ylabel("Brier(augmented) − Brier(la_aug)")
fig.tight_layout(); fig.savefig(f"{F}/fig_r6_blocks.pdf"); plt.close(fig)

# Fig 3: R6 gain by task x mechanism (h = 0.5), diverging around 0
c6 = pd.read_csv(f"{R}/r6/r6_cells.csv"); y = c6[c6.h == .5]
mechs = ["iid", "burst", "source", "incident", "storm", "mixed"]; tasks = ["amf_stall", "ausf_stall", "net_delay", "reg_reject"]
M = np.array([[(y[(y.task == t) & (y.mech == m)].augmented_brier - y[(y.task == t) & (y.mech == m)].la_aug_brier).mean() for m in mechs] for t in tasks])
cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"]); lim = np.abs(M).max()
fig, ax = plt.subplots(figsize=(3.6, 2.0))
im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto"); ax.grid(False)
for i in range(len(tasks)):
    for k in range(len(mechs)):
        ax.text(k, i, f"{M[i, k]:+.4f}", ha="center", va="center", fontsize=6.5, color=INK)
ax.set_xticks(range(len(mechs)), [m + ("\n(control)" if m == "iid" else "") for m in mechs], fontsize=7)
ax.set_yticks(range(len(tasks)), [t.replace("_", " ") for t in tasks], fontsize=7)
cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02); cb.ax.tick_params(labelsize=6); cb.set_label("gain of la_aug", fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/fig_r6_task_mech.pdf"); plt.close(fig)

# Fig 4: availability curve over h
av = j["availability_curve_la_aug_vs_augmented"]; hs = sorted(float(h) for h in av)
est = [av[str(h) if str(h) in av else str(int(h))]["est"] if (str(h) in av or str(int(h)) in av) else np.nan for h in hs]
lo = [av[k]["t_ci95"][0] for k in sorted(av, key=float)]; hi = [av[k]["t_ci95"][1] for k in sorted(av, key=float)]
est = [av[k]["est"] for k in sorted(av, key=float)]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.fill_between(hs, lo, hi, color=C[0], alpha=.15, lw=0); ax.plot(hs, est, color=C[0], lw=2, marker="o", ms=4.5)
ax.axhline(0, color=INK2, lw=.8); ax.set_xlabel("fraction of lineage not recorded, h"); ax.set_ylabel("gain of la_aug (Brier)")
fig.tight_layout(); fig.savefig(f"{F}/fig_r6_availability.pdf"); plt.close(fig)

# Fig 1 of the manuscript: channel comparison. The
# presence curve is a trained invariant representation, not the oracle rule.
# values computed by experiments/paper/figure1_values.py from the frozen cell files (no hand transcription)
g1 = pd.read_csv(f"{R}/editorial/fig1_channel_gaps.csv")
labels = list(g1.setting); count = g1["count"].values; presence = g1.presence.values; negative = g1.negatives.values
fig, ax = plt.subplots(figsize=(7.2, 3.4))
xx = np.arange(len(labels)); width = .24
ax.bar(xx - width, count, width, color=C[0], label="Firing counts, trained GBDT (worst tested mechanism)")
ax.bar(xx, presence, width, color=C[1], label="Presence features, trained invariant GBDT")
ax.bar(xx + width, negative, width, color=C[2], label="Firing counts + non-firing reports, trained GBDT")
ax.set_xticks(xx, labels); ax.set_ylabel("Brier loss relative to learned unique-count fuser")
ax.set_title("Brier loss vs. learned unique-count fuser")
ax.set_ylim(0, .21); ax.legend(fontsize=7.5, loc="upper center", ncol=1)
fig.tight_layout(); fig.savefig(f"{F}/fig_channels.pdf"); plt.close(fig)
print("figures written:", sorted(os.listdir(F)))
