"""R5 confirmatory analysis (docs/r5-protocol.md, frozen before R5 data existed).
Leave-one-block-out over independent testbed blocks; 5 seeds of calibration split and of the
duplication / hidden-lineage randomness. Usage: python -I run_r5.py [features.csv] [out_dir] [--dryrun]"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import (Layout, fit_origin, fuse_generative, loglinear_pool, make_window,  # noqa: E402
                               pact_meet)

ORIGINS = {"REG": ["reg_sig_t90", "reg_log_p90", "reg_ue_p90", "reg_met_failfrac"],
           "AMF": ["amf_sig_react_p90", "amf_met_scrape_err", "amf_log_maxgap"],
           "AUTH": ["auth_sbi_ausf_p90", "auth_sbi_udm_p90", "auth_sbi_udr_p90"],
           "REJ": ["rej_sig", "rej_log", "rej_ue", "rej_sbi404", "rej_met"]}
TASKS = ["amf_stall", "ausf_stall", "reg_reject", "net_delay"]
D_GRID = (0, 1, 2, 4, 8, 16); H_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHAS = np.round(np.linspace(0.05, 1.0, 20), 3)
SEED = 20261026
L = Layout(ORIGINS)
METHODS = ["stacking", "a4", "naive_bayes", "geo_pool", "tb_loglinear", "pact_declared", "pact_true",
           "gen_bma", "gen_oracle"]


def alarms(df, thr):
    return [{o: np.array([int(r[c] > thr[c]) for c in cols]) for o, cols in ORIGINS.items()} for _, r in df.iterrows()]


def main():
    feat = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(ROOT, "results/r5/features.csv")
    out_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else os.path.join(ROOT, "results/r5")
    df = pd.read_csv(feat)
    if "--dryrun" in sys.argv:          # code test only: pseudo-blocks on R4 data
        df = df[~df.pilot].copy(); df["block"] = df.episode % 3 + 1
    else:
        df = df[df.block.isin([1, 2, 3])].copy()
    rows, gate = [], []
    for task in TASKS:
        sub = df[df.label.isin(["healthy", task])].copy()
        if (sub.label == task).sum() == 0:
            continue
        sub["y"] = (sub.label == task).astype(int)
        for held in (1, 2, 3):
            pool, te = sub[sub.block != held], sub[sub.block == held]
            if te.y.nunique() < 2:
                continue
            for seed in range(5):
                rng = np.random.default_rng(SEED + 100 * held + seed + 7 * TASKS.index(task))
                tr_parts, ca_parts = [], []
                for _, g in pool.groupby("y"):
                    idx = rng.permutation(len(g)); a = int(round(0.7 * len(g)))
                    tr_parts.append(g.iloc[idx[:a]]); ca_parts.append(g.iloc[idx[a:]])
                tr, ca = pd.concat(tr_parts), pd.concat(ca_parts)
                thr = {c: float(np.nanpercentile(tr[tr.y == 0][c], 90)) for cols in ORIGINS.values() for c in cols}
                ytr, yca, yte = tr.y.to_numpy(), ca.y.to_numpy(), te.y.to_numpy()
                r1 = np.random.default_rng(SEED * 3 + seed + held)
                trw = [make_window(z, y, 1, 0.0, r1) for z, y in zip(alarms(tr, thr), ytr)]
                caw = [make_window(z, y, 1, 0.0, r1) for z, y in zip(alarms(ca, thr), yca)]
                st = LogisticRegression(C=1.0, max_iter=3000).fit(np.array([L.count_features(w) for w in trw]), ytr)
                a4 = LogisticRegression(C=1.0, max_iter=3000).fit(np.array([L.unit_features(w) for w in trw]), ytr)
                auc_cal = M.auroc(st.predict_proba(np.array([L.count_features(w) for w in caw]))[:, 1], yca)
                gate.append(dict(task=task, held_block=held, seed=seed, auc_cal=auc_cal, passed=bool(auc_cal >= 0.65)))
                models = {o: fit_origin(np.array([a[o] for a in alarms(tr, thr)]), ytr) for o in ORIGINS}
                prior = float(ytr.mean())
                # Taylor-Bishop-type log-linear pooling: alpha tuned on calibration (D_train = 1) by Brier
                alpha = min(ALPHAS, key=lambda a: M.brier([loglinear_pool(w, models, prior, a) for w in caw], yca))
                zte = alarms(te, thr)
                for D in D_GRID:
                    for h in H_GRID:
                        r3 = np.random.default_rng(SEED * 13 + seed * 101 + D * 3 + int(h * 100) + held * 1000 + TASKS.index(task) * 7)
                        P = {m: [] for m in METHODS}; Lo, Up = [], []
                        for z, y in zip(zte, yte):
                            w = make_window(z, y, D, h, r3)
                            P["stacking"].append(float(st.predict_proba(L.count_features(w)[None])[0, 1]))
                            p, lo, up = L.lineage_stacking(w, a4); P["a4"].append(p); Lo.append(lo); Up.append(up)
                            g = fuse_generative(w, models, prior)
                            P["naive_bayes"].append(g["naive_bayes"]); P["gen_bma"].append(g["paef_bma"]); P["gen_oracle"].append(g["oracle"])
                            P["geo_pool"].append(loglinear_pool(w, models, prior, -1))
                            P["tb_loglinear"].append(loglinear_pool(w, models, prior, alpha))
                            P["pact_declared"].append(pact_meet(w, models, prior, "declared"))
                            P["pact_true"].append(pact_meet(w, models, prior, "true"))
                        Lo, Up = np.array(Lo), np.array(Up); dec = (Lo > .5) | (Up < .5)
                        rec = dict(task=task, held_block=held, seed=seed, D=D, h=h, n_test=len(yte), alpha=float(alpha),
                                   cert_share=float(dec.mean()),
                                   cert_err=float(np.mean((Lo[dec] > .5).astype(int) != yte[dec])) if dec.any() else np.nan,
                                   a4_err=float(np.mean((np.array(P["a4"]) > .5).astype(int) != yte)),
                                   a4_width=float(np.mean(Up - Lo)))
                        for m, p in P.items():
                            p = np.array(p)
                            rec[f"{m}_brier"] = M.brier(p, yte); rec[f"{m}_nll"] = M.nll(p, yte)
                            rec[f"{m}_ece"] = M.ece(p, yte); rec[f"{m}_auc"] = M.auroc(p, yte)
                        rows.append(rec)
            print(task, "held", held, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "r5_cells.csv"), index=False)
    pd.DataFrame(gate).to_csv(os.path.join(out_dir, "r5_gate.csv"), index=False)


if __name__ == "__main__":
    main()
