"""Amendment A4 (exploratory, after seeing R2): lineage-aware stacking.
Train a logistic model on countable units (lineage known on training runs, D irrelevant);
at test time known copies are dropped, hidden copies enter with a uniform prior (point) and as an
interval (certificate). Same splits / seeds / windows as run_r2.py."""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
import run_r2 as R  # noqa: E402
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.origami import lineage_stacking, make_window, thresholds, unit_features  # noqa: E402


def main():
    df = pd.read_csv(os.path.join(ROOT, "results/r2/windows_case_study.csv"))
    df = df[df.cores.isin([1, 8])]
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(R.SEED + seed)
        sp = R.split_runs(df, rng)
        for rate in (10, 20, 50, 100):
            tr = sp["train"][sp["train"].rate == rate]; te = sp["test"][sp["test"].rate == rate]
            thr = thresholds(sp["train"], rate)
            ytr = (tr.cores == 1).astype(int).to_numpy()
            r2 = np.random.default_rng(R.SEED * 7 + seed * 31 + rate)
            Xtr = np.array([unit_features(make_window(row.to_dict(), thr, y, 1, 0.0, r2))
                            for (_, row), y in zip(tr.iterrows(), ytr)])
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
            yte = (te.cores == 1).astype(int).to_numpy(); recs = [row.to_dict() for _, row in te.iterrows()]
            for D in (0, 1, 2, 4, 8):
                for h in (0.0, 0.5, 1.0):
                    r3 = np.random.default_rng(R.SEED * 13 + seed * 101 + rate * 7 + D * 3 + int(h * 10))
                    P, L, U = [], [], []
                    for rec, y in zip(recs, yte):
                        o = lineage_stacking(make_window(rec, thr, y, D, h, r3), lr)
                        P.append(o["ls_point"]); L.append(o["ls_lower"]); U.append(o["ls_upper"])
                    P, L, U = map(np.array, (P, L, U))
                    dec = (L > 0.5) | (U < 0.5)
                    rows.append(dict(seed=seed, rate=rate, D=D, h=h, ls_brier=M.brier(P, yte), ls_nll=M.nll(P, yte),
                                     ls_ece=M.ece(P, yte), ls_auc=M.auroc(P, yte), ls_cert_share=float(dec.mean()),
                                     ls_cert_err=float(np.mean((L[dec] > .5).astype(int) != yte[dec])) if dec.any() else np.nan,
                                     ls_width=float(np.mean(U - L))))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/r2/a4_cells.csv"), index=False)


if __name__ == "__main__":
    main()
