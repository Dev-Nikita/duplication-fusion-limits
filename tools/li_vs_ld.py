"""Label-independent versus label-dependent finite-cap game on the training laws of the 108 folds (analytical).
Usage: python3 tools/li_vs_ld.py  -> results/finite_cap/li_vs_ld.csv"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_cvx as C, finite_cap_solvers as S  # noqa: E402

F = pickle.load(open("results/finite_cap/fold_laws.pkl", "rb")); rows = []
for f in F:
    p = f["n_train"] / f["n_train"].sum(); rbu, rbp = S.bayes_risk_U(p), S.bayes_risk_presence(p)
    for D in (1, 2, 4, 8, 16, 19):
        ld, _ = C.ld_minimax(p, D); li, _ = C.li_minimax(p, D)
        rows.append(dict(dataset=f["dataset"], task=f["task"], held_block=f["held_block"], D=D, RB_U=rbu, RB_P=rbp, R_LD=ld, R_LI=li))
d = pd.DataFrame(rows); d.to_csv("results/finite_cap/li_vs_ld.csv", index=False)
d["fracV_LD"] = (d.R_LD - d.RB_U) / (d.RB_P - d.RB_U); d["fracV_LI"] = (d.R_LI - d.RB_U) / (d.RB_P - d.RB_U)
print(d.groupby(["dataset", "task", "D"])[["RB_U", "RB_P", "R_LI", "R_LD", "fracV_LI", "fracV_LD"]].mean().round(3).to_string())
