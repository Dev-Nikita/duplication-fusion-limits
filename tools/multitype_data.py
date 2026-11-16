"""Two report types on the testbed (coordinate-wise copying, label-aware class): what do joint counts retain under a cap
that a single type does not? Types: UE-side slow (20 UEs per window) and AMF-side slow (the UEs whose AMF-side latency
was observed). Law estimated on all blocks of each campaign/condition and task (plug-in; descriptive).
Usage: python3 -W ignore tools/multitype_data.py -> results/finite_cap/multitype.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extended_models as E  # noqa: E402
from apply_finite_cap import TASKS  # noqa: E402

SPECS = [("R10", "results/r10/units.csv", [13, 14, 15, 16, 17, 18], None), ("R6", "results/r6/units.csv", [4, 5, 6, 10, 11, 12], None),
         ("R11-control", "results/r11/units_cond.csv", list(range(21, 33)), "control"), ("R11-loaded", "results/r11/units_cond.csv", list(range(21, 33)), "loaded")]
rows = []
for name, path, blocks, cond in SPECS:
    u = pd.read_csv(path).replace([np.inf], 1e9); u = u[u.block.isin(blocks)]; u = u[u.load_cond == cond] if cond else u
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        t1 = float(np.percentile(sub[sub.y == 0].ue_lat, 90)); seen = sub[(sub.amf_seen == 1)]
        t2 = float(np.percentile(seen[seen.y == 0].amf_lat, 90))
        sub["f1"] = (sub.ue_lat > t1).astype(int); sub["f2"] = ((sub.amf_seen == 1) & (sub.amf_lat > t2)).astype(int)
        w = sub.groupby(["episode", "y"])[["f1", "f2"]].sum().reset_index(); n = len(w)
        def law(cols):
            p = {}
            for y, *uu in zip(w.y, *[w[c] for c in cols]):
                a, b = p.get(tuple(int(v) for v in uu), (0.0, 0.0)); p[tuple(int(v) for v in uu)] = (a + (y == 0) / n, b + (y == 1) / n)
            return p
        laws = {"type1": law(["f1"]), "type2": law(["f2"]), "joint": law(["f1", "f2"])}
        r = dict(dataset=name, task=task, n_windows=n, max_u2=int(w.f2.max()), support_joint=len(laws["joint"]))
        for k, p in laws.items():
            r["RB_U_" + k] = E.bayes(p.values())
            for D in (1, 2, 4):
                v, _ = E.solve_cells(E.cells_label_aware(p, D)); rbp, lo, up = E.multitype_bounds(p, D)
                r[f"Rstar{D}_{k}"] = v; r[f"RB_P_{k}"] = rbp; r[f"Vlo{D}_{k}"] = lo; r[f"Vup{D}_{k}"] = up
                assert lo - 1e-4 <= rbp - v <= up + 1e-4, (name, task, k, D, lo, rbp - v, up)
        rows.append(r); print(name, task, flush=True)
d = pd.DataFrame(rows); d.to_csv("results/finite_cap/multitype.csv", index=False)
pd.set_option("display.width", 250)
print(d[["dataset", "task", "RB_U_type1", "RB_U_type2", "RB_U_joint", "RB_P_type1", "RB_P_joint", "Rstar1_type1", "Rstar1_joint", "Rstar4_type1", "Rstar4_type2", "Rstar4_joint"]].round(3).to_string(index=False))
