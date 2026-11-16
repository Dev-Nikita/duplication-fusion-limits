"""R4 confirmation analysis (docs/r4-protocol.md). Campaign episodes only (pilot excluded)."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import Layout, fit_origin, fuse_generative, make_window  # noqa: E402

ORIGINS = {"REG": ["reg_sig_t90", "reg_log_p90", "reg_ue_p90", "reg_met_failfrac"],
           "AMF": ["amf_sig_react_p90", "amf_met_scrape_err", "amf_log_maxgap"],
           "AUTH": ["auth_sbi_ausf_p90", "auth_sbi_udm_p90", "auth_sbi_udr_p90"],
           "REJ": ["rej_sig", "rej_log", "rej_ue", "rej_sbi404", "rej_met"]}
TASKS = ["amf_stall", "ausf_stall", "reg_reject"]
SEED = 20261024
L = Layout(ORIGINS)


def alarms(df, thr):
    return [{o: np.array([int(r[c] > thr[c]) for c in cols]) for o, cols in ORIGINS.items()} for _, r in df.iterrows()]


def main():
    df = pd.read_csv(os.path.join(ROOT, "results/r4/features.csv"))
    df = df[~df.pilot]
    rows, gate = [], []
    for task in TASKS:
        sub = df[df.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for seed in range(5):
            rng = np.random.default_rng(SEED + seed * 17 + TASKS.index(task))
            parts = {"train": [], "cal": [], "test": []}
            for _, g in sub.groupby("y"):
                idx = rng.permutation(len(g)); n = len(g)
                a, b = int(round(0.5 * n)), int(round(0.7 * n))
                parts["train"].append(g.iloc[idx[:a]]); parts["cal"].append(g.iloc[idx[a:b]]); parts["test"].append(g.iloc[idx[b:]])
            tr, ca, te = (pd.concat(parts[k]) for k in ("train", "cal", "test"))
            healthy = tr[tr.y == 0]
            thr = {c: float(np.nanpercentile(healthy[c], 90)) for cols in ORIGINS.values() for c in cols}
            ytr = tr.y.to_numpy()
            r1 = np.random.default_rng(SEED * 3 + seed)
            trw = [make_window(z, y, 1, 0.0, r1) for z, y in zip(alarms(tr, thr), ytr)]
            st = LogisticRegression(C=1.0, max_iter=3000).fit(np.array([L.count_features(w) for w in trw]), ytr)
            ls = LogisticRegression(C=1.0, max_iter=3000).fit(np.array([L.unit_features(w) for w in trw]), ytr)
            caw = [make_window(z, y, 1, 0.0, np.random.default_rng(seed)) for z, y in zip(alarms(ca, thr), ca.y)]
            auc_cal = M.auroc(st.predict_proba(np.array([L.count_features(w) for w in caw]))[:, 1], ca.y.to_numpy())
            gate.append(dict(task=task, seed=seed, auc_cal=auc_cal, passed=bool(auc_cal >= 0.65)))
            Z = {o: np.array([a[o] for a in alarms(tr, thr)]) for o in ORIGINS}
            models = {o: fit_origin(Z[o], ytr) for o in ORIGINS}
            prior = float(ytr.mean()); zte = alarms(te, thr); yte = te.y.to_numpy()
            for D in (0, 1, 2, 4, 8):
                for h in (0.0, 0.5, 1.0):
                    r3 = np.random.default_rng(SEED * 13 + seed * 101 + D * 3 + int(h * 10) + TASKS.index(task) * 7)
                    P = {k: [] for k in ("stacking", "ls", "oracle", "paef_bma", "naive_bayes")}; Lo, Up = [], []
                    for z, y in zip(zte, yte):
                        w = make_window(z, y, D, h, r3)
                        P["stacking"].append(float(st.predict_proba(L.count_features(w)[None])[0, 1]))
                        p, lo, up = L.lineage_stacking(w, ls); P["ls"].append(p); Lo.append(lo); Up.append(up)
                        for k, v in fuse_generative(w, models, prior).items():
                            P[k].append(v)
                    Lo, Up = np.array(Lo), np.array(Up); dec = (Lo > .5) | (Up < .5)
                    out = dict(task=task, seed=seed, D=D, h=h, n_test=len(yte), cert_share=float(dec.mean()),
                               cert_err=float(np.mean((Lo[dec] > .5).astype(int) != yte[dec])) if dec.any() else np.nan,
                               ls_err=float(np.mean((np.array(P["ls"]) > .5).astype(int) != yte)))
                    for k, p in P.items():
                        p = np.array(p); out[f"{k}_brier"] = M.brier(p, yte); out[f"{k}_auc"] = M.auroc(p, yte); out[f"{k}_ece"] = M.ece(p, yte)
                    rows.append(out)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/r4/r4_cells.csv"), index=False)
    pd.DataFrame(gate).to_csv(os.path.join(ROOT, "results/r4/r4_gate.csv"), index=False)


if __name__ == "__main__":
    main()
