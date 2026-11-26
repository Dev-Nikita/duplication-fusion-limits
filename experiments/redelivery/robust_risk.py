"""Post hoc analysis of the recorded redelivery run (not part of the frozen protocol): for every fitted rule,
  - the worst observed held-out Brier over the within-cap regimes S1, S2, S3a, S4,
  - the exact model worst case on the held-out law of (y, U) over label-aware mechanisms with cap D,
  - the exact model worst case over count-only mechanisms Q(a | u) with cap D,
for D = 0..3 (rules unchanged, fitted for D = 2 under S1), and the observed Brier by held-out physical block.
The rules are refitted exactly as in evaluate.py and checked against the stored results/redelivery/brier.csv.
Usage: python3 -W ignore robust_risk.py confirm/out input/truth.csv ../../tools ../../results/redelivery"""
import os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, sys.argv[3])
import finite_cap_cvx as C, frontier as FR  # noqa: E402

OUT, RES = sys.argv[1], sys.argv[4]; T = pd.read_csv(sys.argv[2]).set_index("episode_id"); D0 = 2; REG = ["S0", "S1", "S2", "S3a", "S3b", "S4"]
A = {r: pd.read_json(os.path.join(OUT, f"deliveries_{r}.jsonl"), lines=True).groupby("episode_id").size().reindex(T.index).fillna(0).astype(int) for r in REG}
stored = pd.read_csv(os.path.join(RES, "brier.csv")).set_index(["dataset", "regime", "rule"]).brier


def law(y, v, m):
    p = np.zeros((2, m + 1))
    for a, b in zip(y, v):
        p[int(a), min(int(b), m)] += 1
    return p / p.sum()


def table(y, v, m):
    n = np.zeros((2, m + 1))
    for a, b in zip(y, v):
        n[int(a), min(int(b), m)] += 1
    eta = pd.Series(np.where(n.sum(0) > 0, (n[1] + 0.5) / (n.sum(0) + 1), np.nan)).ffill().bfill().values
    return lambda q: eta[np.minimum(np.asarray(q, int), m)]


def worst(f, y, u, D, count_only):
    """Exact worst case of the fixed rule f on the empirical law of (y, u): the mechanism picks a in [u, (D+1)u]."""
    tot = 0.0
    for ui in np.unique(u):
        x = np.asarray(f(np.arange(int(ui), (D + 1) * int(ui) + 1)), float); yy = y[u == ui]; n0 = (yy == 0).sum(); n1 = (yy == 1).sum()
        tot += np.max(n0 * x ** 2 + n1 * (1 - x) ** 2) if count_only else n0 * np.max(x ** 2) + n1 * np.max((1 - x) ** 2)
    return tot / len(y)


rows = []; blocks = []; maxdev = 0.0
for ds, Td in T.groupby("dataset"):
    tr = (Td.split == "train").values; te = ~tr; y = Td.y.values; U = Td.U.values; M = int(Td.M.max()); n = (D0 + 1) * M + 1; a1 = A["S1"].loc[Td.index].values
    p = law(y[tr], U[tr], M); q = law(y[tr], a1[tr], n - 1); xm = C.ld_minimax(p, D0)[1]; xw = FR.restricted(q, p, D0, 0.2)
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1, random_state=0).fit(a1[tr].reshape(-1, 1), y[tr])
    pres = table(y[tr], (U[tr] > 0).astype(int), 1); clip = table(y[tr], np.minimum(a1[tr], M), M); lg = table(y[tr], np.floor(2 * np.log2(1 + a1[tr])).astype(int), 16)
    R = {"presence": lambda v: pres((np.asarray(v) > 0).astype(int)), "count_table": table(y[tr], a1[tr], n - 1),
         "count_gbdt": lambda v: gb.predict_proba(np.asarray(v).reshape(-1, 1))[:, 1], "clipped_count": lambda v: clip(np.minimum(v, M)),
         "log_count": lambda v: lg(np.floor(2 * np.log2(1 + np.asarray(v))).astype(int)),
         "minimax": lambda v: xm[np.minimum(np.asarray(v, int), n - 1)], "weighted": lambda v: xw[np.minimum(np.asarray(v, int), n - 1)]}
    for name, f in R.items():
        obs = {}
        for r in REG:
            a = A[r].loc[Td.index].values[te]; b = float(np.mean((f(a) - y[te]) ** 2)); obs[r] = b; maxdev = max(maxdev, abs(b - stored[(ds, r, name)]))
            for blk in sorted(Td.block[te].unique()):
                m = (Td.block.values[te] == blk); blocks.append(dict(dataset=ds, rule=name, regime=r, block=int(blk), n=int(m.sum()), brier=float(np.mean((f(a[m]) - y[te][m]) ** 2))))
        row = dict(dataset=ds, rule=name, worst_observed=max(obs[r] for r in ("S1", "S2", "S3a", "S4")), best_observed=min(obs[r] for r in ("S1", "S2", "S3a", "S4")))
        for D in (0, 1, 2, 3):
            row[f"label_aware_D{D}"] = worst(f, y[te], U[te], D, False); row[f"count_only_D{D}"] = worst(f, y[te], U[te], D, True)
        rows.append(row)
X = pd.DataFrame(rows); X.to_csv(os.path.join(RES, "robust_risk.csv"), index=False); pd.DataFrame(blocks).to_csv(os.path.join(RES, "per_block.csv"), index=False)
out = [f"largest deviation of the refitted observed Brier from the stored brier.csv: {maxdev:.2e}", "",
       X[["dataset", "rule", "worst_observed", "count_only_D2", "label_aware_D2"]].round(4).to_string(index=False), "",
       "sensitivity to the cap (rules unchanged):", X[["dataset", "rule"] + [f"label_aware_D{d}" for d in range(4)] + [f"count_only_D{d}" for d in range(4)]].round(3).to_string(index=False), "",
       "observed Brier by held-out block, worst over S1,S2,S3a,S4:",
       pd.DataFrame(blocks).query("regime in ['S1','S2','S3a','S4']").groupby(["dataset", "rule", "block"]).brier.max().unstack().round(3).to_string()]
open(os.path.join(RES, "robust_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
