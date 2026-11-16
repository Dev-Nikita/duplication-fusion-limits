"""R3b / R3c confirmation with learnability gate (docs/r3-protocol.md)."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.origami import (NODES, STATS, alarms, count_features, fit_origin, fuse, lineage_stacking,  # noqa: E402
                               make_window, unit_features)

SEED = 20261022


def split(df, rng):
    parts = {"train": [], "cal": [], "test": []}
    for _, g in df.groupby(["cores", "rate"]):
        runs = np.array(sorted(g.run.unique())); rng.shuffle(runs)
        for name, sl in (("train", runs[:8]), ("cal", runs[8:14]), ("test", runs[14:20])):
            parts[name].append(g[g.run.isin(sl)])
    return {k: pd.concat(v) for k, v in parts.items()}


def run(pos, neg, tag):
    df = pd.read_csv(os.path.join(ROOT, "results/r2/windows_case_study.csv"))
    df = df[df.cores.isin([pos, neg])].copy(); df["y"] = (df.cores == pos).astype(int)
    rows, gate = [], []
    for seed in range(5):
        sp = split(df, np.random.default_rng(SEED + seed))
        for rate in (10, 20, 50, 100):
            tr = sp["train"][sp["train"].rate == rate]; ca = sp["cal"][sp["cal"].rate == rate]; te = sp["test"][sp["test"].rate == rate]
            nom = tr[tr.y == 0]
            thr = {(n, s): float(np.nanpercentile(nom[f"{n}_{s}"], 90)) for n in NODES for s in STATS}
            ytr = tr.y.to_numpy()
            r2 = np.random.default_rng(SEED * 7 + seed * 31 + rate)
            trw = [make_window(row.to_dict(), thr, y, 1, 0.0, r2) for (_, row), y in zip(tr.iterrows(), ytr)]
            st = LogisticRegression(C=1.0, max_iter=3000).fit(np.array([count_features(w) for w in trw]), ytr)
            # learnability gate on calibration runs
            rg = np.random.default_rng(SEED * 17 + seed * 7 + rate)
            caw = [make_window(row.to_dict(), thr, y, 1, 0.0, rg) for (_, row), y in zip(ca.iterrows(), ca.y.to_numpy())]
            auc_cal = M.auroc(st.predict_proba(np.array([count_features(w) for w in caw]))[:, 1], ca.y.to_numpy())
            gate.append(dict(seed=seed, rate=rate, auc_cal=auc_cal, passed=auc_cal >= 0.65))
            if auc_cal < 0.65:
                continue
            ls = LogisticRegression(C=1.0, max_iter=3000).fit(np.array([unit_features(w) for w in trw]), ytr)
            Z = {n: [] for n in NODES}; Hh = {n: [] for n in NODES}
            for (_, row), y in zip(tr.iterrows(), ytr):
                for n, z in alarms(row.to_dict(), thr).items():
                    Z[n].append(z); Hh[n].append(y)
            models = {n: fit_origin(np.array(Z[n]), np.array(Hh[n])) for n in NODES if len(Z[n]) > 20}
            prior = float(ytr.mean()); yte = te.y.to_numpy(); recs = [row.to_dict() for _, row in te.iterrows()]
            for D in (0, 1, 2, 4, 8):
                for h in (0.0, 0.5, 1.0):
                    r3 = np.random.default_rng(SEED * 13 + seed * 101 + rate * 7 + D * 3 + int(h * 10))
                    P = {k: [] for k in ("stacking", "ls", "oracle", "paef_bma", "naive_bayes")}; L, U = [], []
                    for rec, y in zip(recs, yte):
                        w = make_window(rec, thr, y, D, h, r3)
                        w.z = {n: z for n, z in w.z.items() if n in models}
                        w.copies = [c for c in w.copies if c[0] in models]
                        f = fuse(w, models, prior=prior); o = lineage_stacking(w, ls)
                        P["stacking"].append(float(st.predict_proba(count_features(w)[None])[0, 1]))
                        P["ls"].append(o["ls_point"]); L.append(o["ls_lower"]); U.append(o["ls_upper"])
                        for k in ("oracle", "paef_bma", "naive_bayes"):
                            P[k].append(f[k])
                    L, U = np.array(L), np.array(U); dec = (L > .5) | (U < .5)
                    out = dict(seed=seed, rate=rate, D=D, h=h, cert_share=float(dec.mean()),
                               cert_err=float(np.mean((L[dec] > .5).astype(int) != yte[dec])) if dec.any() else np.nan,
                               ls_err=float(np.mean((np.array(P["ls"]) > .5).astype(int) != yte)))
                    for k, p in P.items():
                        p = np.array(p); out[f"{k}_brier"] = M.brier(p, yte); out[f"{k}_auc"] = M.auroc(p, yte); out[f"{k}_ece"] = M.ece(p, yte)
                    rows.append(out)
        print(tag, "seed", seed, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, f"results/r2/{tag}_cells.csv"), index=False)
    pd.DataFrame(gate).to_csv(os.path.join(ROOT, f"results/r2/{tag}_gate.csv"), index=False)


if __name__ == "__main__":
    run(2, 4, "r3c"); run(2, 8, "r3b")
