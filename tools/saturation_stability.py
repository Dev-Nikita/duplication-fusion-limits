"""Finite-sample stability of the exact saturating cap D_sat versus the quantitative quantities delta_D and
h*delta_D (Proposition 4), by block bootstrap and leave-one-block-out on the full data of each campaign/condition
and task (threshold: 90th percentile of healthy UE latency over all blocks of that campaign/condition).
Analytical, existing data. Usage: python3 -W ignore tools/saturation_stability.py [B] -> results/finite_cap/saturation_stability*.csv"""
import os, sys, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_finite_cap import windows, TASKS, M  # noqa: E402
import saturation_closed_form as SC, quantitative_saturation as QS  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 500; DS = (1, 2, 4, 8, 16); EPS = (0.01, 0.005)
SPECS = [("R10", "results/r10/units.csv", [13, 14, 15, 16, 17, 18], None),
         ("R6", "results/r6/units.csv", [4, 5, 6, 10, 11, 12], None),
         ("R11-control", "results/r11/units_cond.csv", list(range(21, 33)), "control"),
         ("R11-loaded", "results/r11/units_cond.csv", list(range(21, 33)), "loaded")]


def stats(n):
    p = n / n.sum()
    if p[0, 1:].sum() == 0 or p[1, 1:].sum() == 0:
        return None
    r = dict(D_sat=math.ceil(SC.rho(p) - 1e-12) - 1); _, h = QS.constants(p)
    r["h"] = h
    for D in DS:
        d = QS.delta(p, D); r[f"delta_{D}"] = d; r[f"hdelta_{D}"] = h * d; r[f"beta_{D}"] = QS.beta(p, D)
    for e in EPS:   # smallest D on the grid 0..M-1 with h*delta_D <= eps (certified V_D <= eps)
        r[f"D_eps{e}"] = next(D for D in range(0, M) if h * QS.delta(p, D) <= e + 1e-12)
    return r


rng = np.random.default_rng(20261027); full, boot, lobo = [], [], []
for name, path, blocks, cond in SPECS:
    u = pd.read_csv(path).replace([np.inf], 1e9); u = u[u.block.isin(blocks)]
    u = u[u.load_cond == cond] if cond else u
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        thr = float(np.percentile(sub[sub.y == 0].ue_lat, 90)); w = windows(sub, thr)
        per = {b: np.array([[((w.block == b) & (w.y == y) & (np.minimum(w.U, M) == k)).sum() for k in range(M + 1)] for y in (0, 1)])
               for b in blocks}
        r = stats(sum(per.values())); r.update(dataset=name, task=task); full.append(r)
        for i in range(B):
            s = stats(sum(per[b] for b in rng.choice(blocks, size=len(blocks), replace=True)))
            if s:
                s.update(dataset=name, task=task, rep=i); boot.append(s)
        for b in blocks:
            s = stats(sum(per[c] for c in blocks if c != b))
            if s:
                s.update(dataset=name, task=task, dropped=b); lobo.append(s)
        print(name, task, flush=True)
pd.DataFrame(full).to_csv("results/finite_cap/saturation_stability_full.csv", index=False)
pd.DataFrame(boot).to_csv("results/finite_cap/saturation_stability_boot.csv", index=False)
pd.DataFrame(lobo).to_csv("results/finite_cap/saturation_stability_lobo.csv", index=False)
