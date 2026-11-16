"""Summarise the exploratory X4-X7 arm. Usage: python3 -I summarize_explore.py <explore_cells.csv> [more ...]"""
import sys
import numpy as np, pandas as pd
LDM = ("storm_inc", "storm_ben")
for f in sys.argv[1:]:
    c = pd.read_csv(f)
    ref = c[c.method == "gbdt_unit"].set_index(["variant", "task", "held_block", "seed", "mech"])[["brier", "nll"]]
    c = c[c.method != "gbdt_unit"].join(ref, on=["variant", "task", "held_block", "seed", "mech"], rsuffix="_u")
    c["db"] = c.brier - c.brier_u; c["dn"] = c.nll - c.nll_u
    s = c.groupby(["variant", "method", "task", "held_block", "mech"])[["db", "dn"]].mean().reset_index()
    rows = []
    for (v, m), g in s.groupby(["variant", "method"]):
        ld = g[g.mech.isin(LDM)].groupby(["task", "held_block"])[["db", "dn"]].max().groupby("held_block").mean()
        li = g[g.mech.str.startswith("li_")].groupby(["task", "held_block"])[["db", "dn"]].mean().groupby("held_block").mean()
        no = g[g.mech == "none"].groupby("held_block")[["db", "dn"]].mean()
        n = len(ld); t = 2.571 if n == 6 else 2.2
        rows.append(dict(variant=v, method=m, LD_brier=round(ld.db.mean(), 4),
                         LD_ci=f"[{ld.db.mean() - t * ld.db.std() / np.sqrt(n):+.4f},{ld.db.mean() + t * ld.db.std() / np.sqrt(n):+.4f}]",
                         LD_gt_delta=f"{int((ld.db > 0.005).sum())}/{n}", LD_nll=round(ld.dn.mean(), 3),
                         LI_brier=round(li.db.mean(), 4), none_brier=round(no.db.mean(), 4)))
    u = pd.read_csv(f); ub = u[u.method == "gbdt_unit"].groupby("variant").brier.mean().round(4).to_dict()
    print(f, "  gbdt_unit Brier:", ub)
    print(pd.DataFrame(rows).to_string(index=False))
