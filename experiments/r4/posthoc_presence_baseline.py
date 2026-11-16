import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../experiments/r5"))
from sklearn.linear_model import LogisticRegression
from paef.evaluation import metrics as M
from paef.real.generic import make_window
import run_r5 as R
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/r4/features.csv")); df = df[~df.pilot]
out = []
for task in ["amf_stall", "reg_reject"]:
    sub = df[df.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
    for seed in range(5):
        rng = np.random.default_rng(seed); parts = {0: [], 1: []}
        tr, te = [], []
        for _, g in sub.groupby("y"):
            idx = rng.permutation(len(g)); a = int(0.6 * len(g)); tr.append(g.iloc[idx[:a]]); te.append(g.iloc[idx[a:]])
        tr, te = pd.concat(tr), pd.concat(te)
        thr = {c: float(np.nanpercentile(tr[tr.y == 0][c], 90)) for cols in R.ORIGINS.values() for c in cols}
        r1 = np.random.default_rng(seed + 9)
        trw = [make_window(z, y, 1, 0.0, r1) for z, y in zip(R.alarms(tr, thr), tr.y)]
        L = R.L
        st = LogisticRegression(max_iter=3000).fit([L.count_features(w) for w in trw], tr.y)
        a4 = LogisticRegression(max_iter=3000).fit([L.unit_features(w) for w in trw], tr.y)
        pr = LogisticRegression(max_iter=3000).fit([(L.count_features(w) > 0).astype(float) for w in trw], tr.y)
        lg = LogisticRegression(max_iter=3000).fit([np.log1p(L.count_features(w)) for w in trw], tr.y)
        for D, h in [(1, .5), (8, .5), (8, 1.0), (16, 1.0)]:
            r3 = np.random.default_rng(seed * 7 + D + int(h * 10))
            P = {k: [] for k in ["stack", "a4", "presence", "log1p"]}
            for z, y in zip(R.alarms(te, thr), te.y):
                w = make_window(z, y, D, h, r3)
                P["stack"].append(st.predict_proba([L.count_features(w)])[0, 1])
                P["a4"].append(L.lineage_stacking(w, a4)[0])
                P["presence"].append(pr.predict_proba([(L.count_features(w) > 0).astype(float)])[0, 1])
                P["log1p"].append(lg.predict_proba([np.log1p(L.count_features(w))])[0, 1])
            out.append(dict(task=task, seed=seed, D=D, h=h, **{k: M.brier(np.array(v), te.y.to_numpy()) for k, v in P.items()}))
print(pd.DataFrame(out).groupby(["D", "h"])[["stack", "a4", "presence", "log1p"]].mean().round(4))
