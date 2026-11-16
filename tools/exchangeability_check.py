"""Do source identities carry label information beyond the count on the testbed? (firing-set-aware vs count-only
mechanism classes, Proposition on mechanism information). For each campaign/condition and task, leave-one-block-out:
GBDT on [U] versus GBDT on [U, 20 positional firing indicators] (position = rank of the UE's IMSI within the episode;
UEs have no persistent identity across episodes). No duplication is applied. Analytical, existing data.
Usage: python3 -W ignore tools/exchangeability_check.py -> results/finite_cap/exchangeability.csv"""
import os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_finite_cap import TASKS, M  # noqa: E402

SPECS = [("R10", "results/r10/units.csv", [13, 14, 15, 16, 17, 18], None), ("R6", "results/r6/units.csv", [4, 5, 6, 10, 11, 12], None),
         ("R11-control", "results/r11/units_cond.csv", list(range(21, 33)), "control"), ("R11-loaded", "results/r11/units_cond.csv", list(range(21, 33)), "loaded")]
GB = dict(max_depth=3, max_iter=200, learning_rate=0.1, random_state=0); rows = []
for name, path, blocks, cond in SPECS:
    u = pd.read_csv(path).replace([np.inf], 1e9); u = u[u.block.isin(blocks)]; u = u[u.load_cond == cond] if cond else u
    u = u.sort_values(["episode", "imsi"]); u["slot"] = u.groupby("episode").cumcount()
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for b in blocks:
            tr, te = sub[sub.block != b], sub[sub.block == b]
            if te.y.nunique() < 2:
                continue
            thr = float(np.percentile(tr[tr.y == 0].ue_lat, 90))
            def feats(d):
                d = d.assign(f=(d.ue_lat > thr).astype(int)); w = d.pivot_table(index="episode", columns="slot", values="f", fill_value=0)
                w = w.reindex(columns=range(M), fill_value=0); y = d.groupby("episode").y.first().loc[w.index]
                return w.values.sum(1, keepdims=True), np.c_[w.values.sum(1), w.values], y.values
            Utr, Str, ytr = feats(tr); Ute, Ste, yte = feats(te)
            pu = HistGradientBoostingClassifier(**GB).fit(Utr, ytr).predict_proba(Ute)[:, 1]
            ps = HistGradientBoostingClassifier(**GB).fit(Str, ytr).predict_proba(Ste)[:, 1]
            rows.append(dict(dataset=name, task=task, held_block=b, brier_count=np.mean((pu - yte) ** 2), brier_set=np.mean((ps - yte) ** 2), n=len(yte)))
d = pd.DataFrame(rows); d["set_minus_count"] = d.brier_set - d.brier_count; d.to_csv("results/finite_cap/exchangeability.csv", index=False)
print(d.groupby("dataset")[["brier_count", "brier_set", "set_minus_count"]].mean().round(4).to_string())
print("folds where identities help by > 0.005:", int((d.set_minus_count < -0.005).sum()), "of", len(d), "; mean diff", round(d.set_minus_count.mean(), 4))
