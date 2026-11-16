"""Lineage-free baseline rules for the finite-cap comparison (docs/est-protocol.md, Addendum B).
Usage: python3 -W ignore experiments/est/baselines.py results/finite_cap/fold_laws.pkl results/est/baselines.csv"""
import math, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import finite_cap_cvx as C, finite_cap_solvers as S  # noqa: E402

DS = (1, 4, 16)


def eta_table(p):
    s = p.sum(0); e = np.where(s > 0, p[1] / np.where(s > 0, s, 1), np.nan)
    return pd.Series(e).ffill().bfill().fillna(0.5).values


def rules(p, D):
    M = p.shape[1] - 1; A = (D + 1) * M + 1; eta = eta_table(p); mass = p.sum(0); out = {k: np.zeros(A) for k in ("naive", "floor", "mid", "imean", "imid")}
    for a in range(A):
        lo, hi = (0, 0) if a == 0 else (math.ceil(a / (D + 1)), min(a, M))
        out["naive"][a] = eta[min(a, M)]; out["floor"][a] = eta[lo]
        out["mid"][a] = eta[int(min(max(round(2 * a / (D + 2)), lo), hi))]
        w = mass[lo:hi + 1]; e = eta[lo:hi + 1]
        out["imean"][a] = float(np.dot(w, e) / w.sum()) if w.sum() > 0 else float(e.mean())
        out["imid"][a] = 0.5 * (e.min() + e.max())
    e1 = p[1, 1:].sum() / max(p[:, 1:].sum(), 1e-300); pres = np.full(A, e1); pres[0] = eta[0]; out["presence"] = pres
    out["minimax"] = C.ld_minimax(p, D)[1]
    return out


F = pickle.load(open(sys.argv[1], "rb")); rows = []
for f in F:
    ptr = f["n_train"] / f["n_train"].sum(); pte = f["n_test"] / f["n_test"].sum()
    for D in DS:
        r = dict(dataset=f["dataset"], task=f["task"], held_block=f["held_block"], D=D)
        for k, x in rules(ptr, D).items():
            r["W_" + k] = S.primal_value(x, pte, D); r["nocopy_" + k] = S.primal_value(x, pte, 0)
        rows.append(r)
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False)
K = ["naive", "floor", "mid", "imean", "imid", "presence", "minimax"]
print(d.groupby("D")[["W_" + k for k in K]].mean().round(4).to_string()); print(d.groupby("D")[["nocopy_" + k for k in K]].mean().round(4).to_string())
for D, z in d.groupby("D"):
    print(D, {k: int((z.W_minimax < z["W_" + k] - 1e-12).sum()) for k in K if k != "minimax"})
