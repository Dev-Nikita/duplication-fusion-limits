"""R5b analysis (docs/r5b-protocol.md): per-UE unit layer, leave-one-block-out, strong lineage-free
baselines. Usage: python3 -I run_r5b.py <units.csv> <out_dir> [--dryrun]"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import fit_origin, loglinear_pool, make_window, pact_meet, _llr  # noqa: E402
from paef.real.units import UnitLayout  # noqa: E402

TASKS = ["amf_stall", "ausf_stall", "reg_reject", "net_delay"]
D_GRID = (0, 1, 2, 4, 8, 16); H_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
AUG_D = (0, 1, 2, 4, 8)                     # D = 16 is never seen by the augmented model
ALPHAS = np.round(np.linspace(0.05, 1.0, 20), 3)
SEED = 20261027
K = 4                                        # readings per UE: ue_slow (primary), amf_slow, ue_rej, amf_rej
L = UnitLayout(K)
METHODS = ["stacking", "a4", "presence", "log1p", "augmented", "matched", "naive_bayes", "tb_loglinear",
           "pact_declared", "pact_true"]


def episodes(units, thr):
    out = []
    for ep, g in units.groupby("episode", sort=True):
        z = {r.imsi: np.array([int(r.ue_lat > thr[0]), int(r.amf_lat > thr[1]), int(r.ue_rej), int(r.amf_rej)])
             for r in g.itertuples()}
        out.append((ep, int(g.y.iloc[0]), z))
    return out


def fit(X, y):
    return LogisticRegression(C=1.0, max_iter=5000).fit(np.array(X), np.array(y))


def naive(w, m, prior):
    s = np.log(prior / (1 - prior))
    for zz in w.z.values():
        s += sum(_llr(m, k, zz[k]) for k in range(len(zz)))
    s += sum(_llr(m, k, 1) for _, k, _ in w.copies)
    return float(1 / (1 + np.exp(-s)))


def main():
    units = pd.read_csv(sys.argv[1]); out_dir = sys.argv[2]; dry = "--dryrun" in sys.argv
    units = units.replace([np.inf], 1e9)
    if dry:
        units = units[(units.block.isna()) & (~units.pilot)].copy(); units["block"] = units.episode % 3 + 1
    else:
        units = units[units.block.isin([1, 2, 3])].copy()
        bad = units[(units.label == "net_delay") & (units.netem_ok != True)].episode.unique()  # noqa: E712
        units = units[~units.episode.isin(bad)]
        dup = units.drop_duplicates("episode").duplicated(["block", "sched_pos"])
        units = units[~units.episode.isin(units.drop_duplicates("episode")[dup].episode)]
    rows, gate = [], []
    for task in TASKS:
        sub = units[units.label.isin(["healthy", task])].copy()
        if (sub.label == task).sum() == 0:
            continue
        sub["y"] = (sub.label == task).astype(int)
        for held in (1, 2, 3):
            pool, te = sub[sub.block != held], sub[sub.block == held]
            if te.y.nunique() < 2:
                continue
            for seed in range(5):
                rng = np.random.default_rng(SEED + 100 * held + seed + 7 * TASKS.index(task))
                eps = pool.drop_duplicates("episode")[["episode", "y"]]
                tr_e, ca_e = [], []
                for _, g in eps.groupby("y"):
                    idx = rng.permutation(len(g)); a = int(round(0.7 * len(g)))
                    tr_e += list(g.episode.iloc[idx[:a]]); ca_e += list(g.episode.iloc[idx[a:]])
                tr, ca = pool[pool.episode.isin(tr_e)], pool[pool.episode.isin(ca_e)]
                h0 = tr[tr.y == 0]
                thr = (float(np.percentile(h0.ue_lat, 90)), float(np.percentile(h0.amf_lat, 90)))
                E_tr, E_ca, E_te = episodes(tr, thr), episodes(ca, thr), episodes(te, thr)
                ytr = [y for _, y, _ in E_tr]; yca = np.array([y for _, y, _ in E_ca]); yte = np.array([y for _, y, _ in E_te])
                r1 = np.random.default_rng(SEED * 3 + seed + held)
                W1 = [make_window(z, y, 1, 0.0, r1) for _, y, z in E_tr]
                st = fit([L.count_features(w) for w in W1], ytr)
                a4 = fit([L.unit_features(w) for w in W1], ytr)
                pr = fit([L.presence_features(w) for w in W1], ytr)
                lg = fit([L.log_features(w) for w in W1], ytr)
                Waug, yaug = [], []
                for rep in range(5):
                    for _, y, z in E_tr:
                        Waug.append(make_window(z, y, int(r1.choice(AUG_D)), float(r1.choice(H_GRID)), r1)); yaug.append(y)
                au = fit([L.count_features(w) for w in Waug], yaug)
                Wca = [make_window(z, y, 1, 0.0, r1) for _, y, z in E_ca]
                auc_cal = M.auroc(st.predict_proba(np.array([L.count_features(w) for w in Wca]))[:, 1], yca)
                gate.append(dict(task=task, held_block=held, seed=seed, auc_cal=auc_cal, passed=bool(auc_cal >= 0.65)))
                Z = np.array([zz for _, _, z in E_tr for zz in z.values()])
                H = np.array([y for _, y, z in E_tr for _ in z])
                m = fit_origin(Z, H); prior = float(np.mean(ytr))
                alpha = min(ALPHAS, key=lambda a: M.brier([loglinear_pool(w, {o: m for o in w.z}, prior, a) for w in Wca], yca))
                for D in D_GRID:
                    for h in H_GRID:
                        r3 = np.random.default_rng(SEED * 13 + seed * 101 + D * 3 + int(h * 100) + held * 1000 + TASKS.index(task) * 7)
                        Wm = [make_window(z, y, D, h, r3) for _, y, z in E_tr]
                        ma = fit([L.count_features(w) for w in Wm], ytr)
                        P = {k: [] for k in METHODS}; Lo, Up = [], []
                        for _, y, z in E_te:
                            w = make_window(z, y, D, h, r3); models = {o: m for o in w.z}
                            c = L.count_features(w)[None]
                            P["stacking"].append(st.predict_proba(c)[0, 1])
                            p, lo, up = L.lineage_stacking(w, a4); P["a4"].append(p); Lo.append(lo); Up.append(up)
                            P["presence"].append(pr.predict_proba(L.presence_features(w)[None])[0, 1])
                            P["log1p"].append(lg.predict_proba(L.log_features(w)[None])[0, 1])
                            P["augmented"].append(au.predict_proba(c)[0, 1])
                            P["matched"].append(ma.predict_proba(c)[0, 1])
                            P["naive_bayes"].append(naive(w, m, prior))
                            P["tb_loglinear"].append(loglinear_pool(w, models, prior, alpha))
                            P["pact_declared"].append(pact_meet(w, models, prior, "declared"))
                            P["pact_true"].append(pact_meet(w, models, prior, "true"))
                        Lo, Up = np.array(Lo), np.array(Up); dec = (Lo > .5) | (Up < .5)
                        rec = dict(task=task, held_block=held, seed=seed, D=D, h=h, n_test=len(yte), alpha=float(alpha),
                                   dec_share=float(dec.mean()),
                                   dec_err=float(np.mean((Lo[dec] > .5).astype(int) != yte[dec])) if dec.any() else np.nan,
                                   a4_err=float(np.mean((np.array(P["a4"]) > .5).astype(int) != yte)),
                                   a4_width=float(np.mean(Up - Lo)))
                        for k, p in P.items():
                            p = np.array(p)
                            rec[f"{k}_brier"] = M.brier(p, yte); rec[f"{k}_nll"] = M.nll(p, yte)
                            rec[f"{k}_ece"] = M.ece(p, yte); rec[f"{k}_auc"] = M.auroc(p, yte)
                        rows.append(rec)
            print(task, "held", held, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "r5b_cells.csv"), index=False)
    pd.DataFrame(gate).to_csv(os.path.join(out_dir, "r5b_gate.csv"), index=False)


if __name__ == "__main__":
    main()
