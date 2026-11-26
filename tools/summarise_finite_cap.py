"""Summarise the finite-cap application (analytical; no new data). Writes results/finite_cap/summary.csv,
results/finite_cap/saturation.csv (the figure is drawn by tools/paper_figures.py). Usage: python3 tools/summarise_finite_cap.py"""
import glob
import numpy as np, pandas as pd

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
