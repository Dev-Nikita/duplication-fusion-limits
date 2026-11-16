"""Summarise the finite-cap application (analytical; no new data). Writes results/finite_cap/summary.csv,
results/finite_cap/saturation.csv and paper/figures/fig_copycap.pdf. Usage: python3 tools/summarise_finite_cap.py"""
import glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/finite_cap/"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "part*/finite_cap_application.csv"))], ignore_index=True)
pr = pd.read_csv(R + "presence_rule_oos.csv")
d = d.merge(pr, on=["dataset", "task", "held_block"], how="left")
cv = pd.read_csv(R + "plugin_cvx_oos.csv")
d = d.merge(cv, on=["dataset", "task", "held_block", "D"], how="left")
# primary out-of-sample rule: interior-point solution; mirror-ascent rule kept as sensitivity check
d = d.rename(columns={"W_test_minimax": "W_test_mirror", "nocopy_test_minimax": "nocopy_test_mirror"})
d["W_test_minimax"] = d.W_test_cvx; d["nocopy_test_minimax"] = d.nocopy_test_cvx
d["gap"] = d.R_star_up - d.R_star_lo
d["cvx_minus_mirror"] = (d.R_star_cvx - d.R_star_up).abs()  # interior-point vs mirror-ascent upper bound
d["sat_exact"] = d.saturated_lp
s = d.groupby(["dataset", "task", "D"]).agg(
    folds=("held_block", "size"), R_star=("R_star_cvx", "mean"), R_star_min=("R_star_cvx", "min"), R_star_max=("R_star_cvx", "max"),
    R_star_mirror_up=("R_star_up", "mean"), max_abs_cvx_minus_mirror=("cvx_minus_mirror", "max"),
    RB_U=("RB_U", "mean"), RB_P=("RB_P", "mean"), sat_exact=("sat_exact", "mean"), max_gap=("gap", "max"),
    W_minimax=("W_test_minimax", "mean"), W_naive=("W_test_naive", "mean"), presence_test=("presence_rule_test", "mean"),
    nocopy_minimax=("nocopy_test_minimax", "mean"), nocopy_naive=("nocopy_test_naive", "mean"),
    W_mirror=("W_test_mirror", "mean")).reset_index()
s["frac_of_V"] = (s.R_star - s.RB_U) / (s.RB_P - s.RB_U).replace(0, np.nan)
s["minimax_beats_naive_folds"] = d.assign(w=d.W_test_minimax < d.W_test_naive - 1e-12).groupby(["dataset", "task", "D"]).w.sum().values
s["minimax_beats_presence_folds"] = d.assign(w=d.W_test_minimax < d.presence_rule_test - 1e-12).groupby(["dataset", "task", "D"]).w.sum().values
s["mirror_beats_presence_folds"] = d.assign(w=d.W_test_mirror < d.presence_rule_test - 1e-12).groupby(["dataset", "task", "D"]).w.sum().values
for D, z in d.groupby("D"):
    print(f"D={D}: minimax<naive {int((z.W_test_minimax < z.W_test_naive - 1e-12).sum())}/{len(z)}, "
          f"minimax<presence {int((z.W_test_minimax < z.presence_rule_test - 1e-12).sum())}, mirror<presence "
          f"{int((z.W_test_mirror < z.presence_rule_test - 1e-12).sum())}, median naive-minimax "
          f"{(z.W_test_naive - z.W_test_minimax).median():.3f}, nocopy cost median "
          f"{(z.nocopy_test_minimax - z.nocopy_test_naive).median():.4f} max {(z.nocopy_test_minimax - z.nocopy_test_naive).max():.3f}")
s.round(4).to_csv(R + "summary.csv", index=False)
# smallest saturating cap per fold (LP-exact), over the D grid
# smallest saturating cap ON THE GRID {1,2,3,4,8,16} (LP test). "not reached" means no saturation for any grid D <= 16,
# NOT that saturation never occurs: the exact cap D_sat (closed form, always <= M-1 = 19) is in saturation_closed_form.csv.
sat = d[d.sat_exact & (d.D > 0)].groupby(["dataset", "task", "held_block"]).D.min().rename("D_sat_grid")
allf = d[["dataset", "task", "held_block"]].drop_duplicates().set_index(["dataset", "task", "held_block"])
allf = allf.join(sat); allf["reached_on_grid"] = allf.D_sat_grid.notna()
allf.reset_index().to_csv(R + "saturation.csv", index=False)
print("smallest saturating cap on the grid {1,2,3,4,8,16} per fold ('-' = not reached up to 16; exact D_sat in saturation_closed_form.csv):")
print(allf.reset_index().groupby(["dataset", "task"]).D_sat_grid.agg(lambda x: " ".join("-" if pd.isna(v) else str(int(v)) for v in x)).to_string())
print("max primal-dual gap", d.gap.max(), " value/LP saturation contradictions",
      int(((d.saturated_lp) & (d.R_star_lo < d.RB_P - 1e-4)).sum()))
# figure
ds = list(dict.fromkeys(d.dataset)); tasks = ["amf_stall", "ausf_stall", "net_delay"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
cols = dict(zip(ds, ["#1f5aa6", "#c2571a", "#2a8a4a", "#7b3fa0"]))
for ax, t in zip(axs, tasks):
    for k in ds:
        z = s[(s.dataset == k) & (s.task == t)]
        if z.empty:
            continue
        xs = np.arange(len(z))
        ax.plot(xs, z.R_star, "-o", ms=3, color=cols[k], label=k.replace("-control", " ctrl").replace("-loaded", " load"))
        ax.plot(xs, z.W_minimax, ":", color=cols[k], lw=1)
        ax.axhline(z.RB_P.iloc[0], color=cols[k], lw=0.6, alpha=0.5, ls="--")
    ax.set_xticks(np.arange(len(z))); ax.set_xticklabels([str(v) for v in z.D]); ax.set_title(t.replace("_", " "), fontsize=9)
    ax.set_xlabel("copy cap $D$", fontsize=8); ax.tick_params(labelsize=7.5); ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylabel("Brier risk", fontsize=8); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, frameon=False, loc="upper center", ncol=4)
import os; os.makedirs("paper/figures", exist_ok=True)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("paper/figures/fig_copycap.pdf"); print("figure written")
