"""Out-of-sample Brier risk of the presence rule (x = P_train(y=1 | U>0) for every positive count; invariant under
every admissible duplication, so it is also its worst case). Same folds and thresholds as apply_finite_cap.py.
Usage: python3 presence_rule_oos.py <out.csv> <dataset:units.csv:blocks[:cond]> ..."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_finite_cap import law, windows, TASKS  # noqa: E402

rows = []
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
            ptr, pte = law(windows(tr, thr)), law(windows(te, thr))
            e0 = ptr[1, 0] / ptr[:, 0].sum() if ptr[:, 0].sum() > 0 else 0.5
            e1 = ptr[1, 1:].sum() / ptr[:, 1:].sum() if ptr[:, 1:].sum() > 0 else 0.5
            r = pte[0, 0] * e0 ** 2 + pte[1, 0] * (1 - e0) ** 2 + pte[0, 1:].sum() * e1 ** 2 + pte[1, 1:].sum() * (1 - e1) ** 2
            rows.append(dict(dataset=name, task=task, held_block=b, presence_rule_test=r))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
