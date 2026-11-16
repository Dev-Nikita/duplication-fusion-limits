"""Detection models on RCAEval window features (docs/r12-protocol.md).
Each case gives two test windows (N: label 0, F: label 1), each described relative to the case's history window H:
feature = log((x_W + 1) / (x_H + 1)) for the raw quantities of the feature set.
Feature sets (channels):
  COUNT  {A}                       message counts, copies included
  UNIT   {U, P, T}                 distinct-unit statistics (copies of a message from one pod removed)
  BOTH   {A, U, P, T}              counts plus distinct units
  RATE   {A/L, U/K}                denominator-normalised (non-error lines and reporting pods as denominators)
Learners: logistic regression (L2, C = 1, standardised) and GBDT (HistGradientBoosting, depth 2, 100 iterations,
lr 0.1, min_samples_leaf 5); fixed hyperparameters, no tuning.
Schemes:
  within : 5-fold cross-validation within each system, folds by fault_dir (service x fault type), so the
           replicates of one injected fault never sit on both sides
  cross  : train on two systems, test on the third
Output: one row per (scheme, learner, feature set, test case) with the case Brier (mean over its N and F windows)
and the window scores. Usage: python3 -I rca_model.py <features.csv> <out.csv>"""
import sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SETS = {"COUNT": ["A"], "UNIT": ["U", "P", "T"], "BOTH": ["A", "U", "P", "T"], "RATE": ["AL", "UK"]}


def windows(df):
    df = df.copy(); df["AL"] = df.A / df.L.clip(lower=1); df["UK"] = df.U / df.K.clip(lower=1)
    h = df[df.window == "H"].set_index("case")
    out = []
    for w, y in (("N", 0), ("F", 1)):
        x = df[df.window == w].set_index("case")
        r = pd.DataFrame(index=x.index)
        for c in ("A", "U", "P", "T"):
            r[c] = np.log((x[c] + 1) / (h.loc[x.index, c] + 1))
        for c in ("AL", "UK"):
            r[c] = np.log((x[c] + 1e-3) / (h.loc[x.index, c] + 1e-3))
        r["y"] = y; r["system"] = x.system; r["fault_dir"] = x.fault_dir; r["window"] = w
        out.append(r.reset_index())
    return pd.concat(out, ignore_index=True)


def learners():
    return {"logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
            "gbdt": lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=0.1, min_samples_leaf=5, random_state=0)}


def fit_predict(tr, te, cols, mk):
    if tr.y.nunique() < 2:
        return np.full(len(te), tr.y.mean())
    return mk().fit(tr[cols].values, tr.y.values).predict_proba(te[cols].values)[:, 1]


def main():
    W = windows(pd.read_csv(sys.argv[1])); rows = []
    systems = sorted(W.system.unique())
    splits = []
    for s in systems:
        Ws = W[W.system == s]
        k = min(5, Ws.fault_dir.nunique())
        for i, (a, b) in enumerate(GroupKFold(n_splits=k).split(Ws, groups=Ws.fault_dir)):
            splits.append(("within", s, Ws.iloc[a], Ws.iloc[b]))
    for s in systems:
        splits.append(("cross", s, W[W.system != s], W[W.system == s]))
    for scheme, s, tr, te in splits:
        for ln, mk in learners().items():
            for fs, cols in SETS.items():
                p = fit_predict(tr, te, cols, mk)
                t = te.assign(p=p, se=(p - te.y.values) ** 2)
                for case, g in t.groupby("case"):
                    rows.append(dict(scheme=scheme, test_system=s, learner=ln, set=fs, case=case,
                                     fault_dir=g.fault_dir.iloc[0], brier=g.se.mean(),
                                     p_N=float(g[g.window == "N"].p.iloc[0]) if (g.window == "N").any() else np.nan,
                                     p_F=float(g[g.window == "F"].p.iloc[0]) if (g.window == "F").any() else np.nan))
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
    main()
