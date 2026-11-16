"""Held-out exact worst-case risk of the plug-in finite-cap rule computed with the interior-point solver
(finite_cap_cvx.ld_minimax), for every fold and every D of the grid. The finite-cap minimax rule is not unique
(counts without training mass, flat regions); this is the primary implementation, the mirror-ascent rule of
apply_finite_cap.py is kept as a sensitivity check. Usage: python3 tools/plugin_cvx_oos.py"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_cvx as C, finite_cap_solvers as S  # noqa: E402

F = pickle.load(open("results/finite_cap/fold_laws.pkl", "rb")); rows = []
for f in F:
    ptr = f["n_train"] / f["n_train"].sum(); pte = f["n_test"] / f["n_test"].sum()
    for D in (0, 1, 2, 3, 4, 8, 16):
        v, x = C.ld_minimax(ptr, D)
        rows.append(dict(dataset=f["dataset"], task=f["task"], held_block=f["held_block"], D=D, R_star_cvx=v,
                         W_test_cvx=S.primal_value(x, pte, D), nocopy_test_cvx=S.primal_value(x, pte, 0)))
pd.DataFrame(rows).to_csv("results/finite_cap/plugin_cvx_oos.csv", index=False); print(len(rows))
