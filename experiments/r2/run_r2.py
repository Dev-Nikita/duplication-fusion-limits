"""R2 run (protocol docs/r2-protocol.md, frozen before this run)."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from paef.evaluation import metrics as M  # noqa: E402
from paef.real.origami import NODES, alarms, count_features, fit_origin, fuse, make_window, thresholds  # noqa: E402

SEED = 20261020
METHODS = ["naive_bayes", "geo_pool", "stacking", "pact_meet_declared", "declared_only", "paef_bma",
           "pact_meet_closure", "oracle"]


def split_runs(df, rng):
    parts = {"train": [], "cal": [], "test": []}
    for (c, r), g in df.groupby(["cores", "rate"]):
        runs = np.array(sorted(g.run.unique()))
        rng.shuffle(runs)
        for name, sl in (("train", runs[:8]), ("cal", runs[8:14]), ("test", runs[14:20])):
            parts[name].append(g[g.run.isin(sl)])
    return {k: pd.concat(v) for k, v in parts.items()}


def main():
    df = pd.read_csv(os.path.join(ROOT, "results/r2/windows_case_study.csv"))
    df = df[df.cores.isin([1, 8])]
    rows = []
    t0 = time.time()
    for seed in range(5):
        rng = np.random.default_rng(SEED + seed)
        sp = split_runs(df, rng)
        for rate in (10, 20, 50, 100):
            tr = sp["train"][sp["train"].rate == rate]
            te = sp["test"][sp["test"].rate == rate]
            thr = thresholds(sp["train"], rate)
            ytr = (tr.cores == 1).astype(int).to_numpy()
            Ztr = {n: [] for n in NODES}; Htr = {n: [] for n in NODES}
            for (_, row), y in zip(tr.iterrows(), ytr):
                for n, z in alarms(row.to_dict(), thr).items():
                    Ztr[n].append(z); Htr[n].append(y)
            models = {n: fit_origin(np.array(Ztr[n]), np.array(Htr[n])) for n in NODES if len(Ztr[n]) > 20}
            # stacking trained at D_train = 1 (lineage-blind)
            r2 = np.random.default_rng(SEED * 7 + seed * 31 + rate)
            Xtr = np.array([count_features(make_window(row.to_dict(), thr, y, 1, 0.0, r2))
                            for (_, row), y in zip(tr.iterrows(), ytr)])
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
            yte = (te.cores == 1).astype(int).to_numpy()
            recs = [row.to_dict() for _, row in te.iterrows()]
            for D in (0, 1, 2, 4, 8):
                for h in (0.0, 0.5, 1.0):
                    r3 = np.random.default_rng(SEED * 13 + seed * 101 + rate * 7 + D * 3 + int(h * 10))
                    out = {m: [] for m in METHODS}; lo, up = [], []
                    for rec, y in zip(recs, yte):
                        w = make_window(rec, thr, y, D, h, r3)
                        w.z = {n: z for n, z in w.z.items() if n in models}
                        w.copies = [c for c in w.copies if c[0] in models]
                        f = fuse(w, models)
                        for m in METHODS:
                            if m != "stacking":
                                out[m].append(f[m])
                        out["stacking"].append(float(lr.predict_proba(count_features(w)[None])[0, 1]))
                        lo.append(f["paef_lower"]); up.append(f["paef_upper"])
                    lo, up = np.array(lo), np.array(up)
                    dec = (lo > 0.5) | (up < 0.5)
                    rec_out = dict(seed=seed, rate=rate, D=D, h=h, n=len(yte),
                                   cert_share=float(dec.mean()),
                                   cert_err=float(np.mean((lo[dec] > 0.5).astype(int) != yte[dec])) if dec.any() else np.nan,
                                   width=float(np.mean(up - lo)))
                    for m in METHODS:
                        p = np.array(out[m])
                        rec_out[f"{m}_brier"] = M.brier(p, yte); rec_out[f"{m}_nll"] = M.nll(p, yte)
                        rec_out[f"{m}_ece"] = M.ece(p, yte); rec_out[f"{m}_auc"] = M.auroc(p, yte)
                    rows.append(rec_out)
            print("seed", seed, "rate", rate, round(time.time() - t0), flush=True)
    os.makedirs(os.path.join(ROOT, "results/r2"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/r2/r2_cells.csv"), index=False)


if __name__ == "__main__":
    main()
