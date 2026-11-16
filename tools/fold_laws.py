"""Dump the per-fold empirical evidence laws used by apply_finite_cap.py (same folds, thresholds, report type),
including the per-block laws of the training blocks (for inner-block analyses).
Usage: python3 tools/fold_laws.py <out.npz> <dataset:units.csv:blocks[:cond]> ..."""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_finite_cap import law, windows, TASKS  # noqa: E402

folds = []
for spec in sys.argv[2:]:
    parts = spec.split(":"); name, path, blocks = parts[0], parts[1], [int(b) for b in parts[2].split(",")]
    cond = parts[3] if len(parts) > 3 else None
    u = pd.read_csv(path).replace([np.inf], 1e9); u = u[u.block.isin(blocks)]
    if cond:
        u = u[u.load_cond == cond]
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for b in blocks:
            tr, te = sub[sub.block != b], sub[sub.block == b]
            if te.y.nunique() < 2:
                continue
            thr = float(np.percentile(tr[tr.y == 0].ue_lat, 90))
            wtr = windows(tr, thr); wte = windows(te, thr)
            counts = lambda w: np.array([[((w.y == y) & (np.minimum(w.U, 20) == k)).sum() for k in range(21)] for y in (0, 1)])
            per_block = {int(bb): counts(wtr[wtr.block == bb]) for bb in sorted(wtr.block.unique())}
            folds.append(dict(dataset=name, task=task, held_block=b, thr=thr, n_train=counts(wtr), n_test=counts(wte),
                              per_block=per_block))
with open(sys.argv[1], "wb") as f:
    pickle.dump(folds, f)
print(len(folds), "folds")
