"""Cross-platform numerical check: recompute interior-point finite-cap values and held-out worst-case risks for a slice of
the fold laws and compare with the stored reference (results/finite_cap/plugin_cvx_oos.csv). Also checks that each value
lies inside the stored, independently computed mirror-ascent bracket (primal/dual certificate).
Tolerance fixed in advance: 1e-4 on values (solver tolerance stated in REPRODUCE.md); 5e-3 on held-out risks of the rule
(the minimax rule is not unique, so the rule itself may differ across platforms while the value does not).
Usage: python3 -W ignore tools/cross_platform_check.py START STOP   (fold indices, 0..108)"""
import glob, os, pickle, platform, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_cvx as C, finite_cap_solvers as S  # noqa: E402

a, b = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
F = pickle.load(open("results/finite_cap/fold_laws.pkl", "rb"))[a:b]
ref = pd.read_csv("results/finite_cap/plugin_cvx_oos.csv").set_index(["dataset", "task", "held_block", "D"])
mir = pd.concat([pd.read_csv(f) for f in glob.glob("results/finite_cap/part*/finite_cap_application.csv")]).set_index(["dataset", "task", "held_block", "D"])
dv, dw, out_br = [], [], 0
for f in F:
    ptr = f["n_train"] / f["n_train"].sum(); pte = f["n_test"] / f["n_test"].sum()
    for D in (0, 1, 2, 3, 4, 8, 16):
        k = (f["dataset"], f["task"], f["held_block"], D); v, x = C.ld_minimax(ptr, D)
        dv.append(abs(v - ref.loc[k, "R_star_cvx"])); dw.append(abs(S.primal_value(x, pte, D) - ref.loc[k, "W_test_cvx"]))
        out_br += not (mir.loc[k, "R_star_lo"] - 1e-4 <= v <= mir.loc[k, "R_star_up"] + 1e-4)
        assert abs(S.primal_value(x, ptr, D) - v) < 1e-4, "returned rule does not attain the value"
import cvxpy, scipy
print(f"{platform.system()} {platform.machine()} py{platform.python_version()} numpy{np.__version__} scipy{scipy.__version__} cvxpy{cvxpy.__version__} | "
      f"folds {a}:{b} cells {len(dv)} | max|dR*| {max(dv):.2e} | max|dW| {max(dw):.2e} | outside bracket {out_br} | {time.time() - t0:.0f}s")
