"""Certify the count-only (label-independent) game values: mirror ascent on its dual (shared kernel Q(a|u) for both
classes) gives a LOWER bound; the interior-point rule's exact worst case (li_worst_case) gives an UPPER bound.
Usage: python3 -W ignore tools/li_certify.py -> results/finite_cap/li_certificate.csv"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_cvx as C, finite_cap_solvers as S  # noqa: E402


def li_dual_lower(p, D, iters=6000, eta=4.0):
    M = p.shape[1] - 1; I = S.intervals(M, D); A = (D + 1) * M + 1
    Q = [np.full(len(I[u]), 1.0 / len(I[u])) for u in range(M + 1)]; best = -1.0
    for t in range(iters):
        w0 = np.zeros(A); w1 = np.zeros(A)
        for u in range(M + 1):
            w0[I[u]] += p[0, u] * Q[u]; w1[I[u]] += p[1, u] * Q[u]
        s = w0 + w1; m = s > 0; val = float(np.sum(w0[m] * w1[m] / s[m])); best = max(best, val)
        x = np.where(m, w1 / np.where(m, s, 1), 0.5)
        step = eta / np.sqrt(t + 1)
        for u in range(M + 1):
            g = p[0, u] * x[I[u]] ** 2 + p[1, u] * (1 - x[I[u]]) ** 2
            Q[u] = Q[u] * np.exp(step * (g - g.max()) / max(p[:, u].sum(), 1e-12)); Q[u] /= Q[u].sum()
    return best


F = pickle.load(open("results/finite_cap/fold_laws.pkl", "rb")); rows = []
for f in F:
    p = f["n_train"] / f["n_train"].sum()
    for D in (1, 2, 4, 8, 16, 19):
        v, x = C.li_minimax(p, D); up = C.li_worst_case(x, p, D); lo = li_dual_lower(p, D)
        rows.append(dict(dataset=f["dataset"], task=f["task"], held_block=f["held_block"], D=D, li_lower=lo, li_upper=up))
    print(f["dataset"], f["task"], f["held_block"], flush=True)
d = pd.DataFrame(rows); d["gap"] = d.li_upper - d.li_lower; d.to_csv("results/finite_cap/li_certificate.csv", index=False)
print(d.gap.describe())
