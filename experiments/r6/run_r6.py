"""R6 analysis (docs/r6-protocol.md): duplication-MECHANISM shift. Training-time augmentation sees only
iid duplication; test windows use iid (control) and five unseen mechanisms. Per-UE unit layer of R5b
(build_units.py, unchanged). Leave-one-block-out over blocks 4-9.
Usage: python3 -I run_r6.py <units.csv> <out_dir> [--dryrun]"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "r5b"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import fit_origin, fuse_generative, loglinear_pool, make_window, pact_meet  # noqa: E402
from paef.real.mechanisms import UNSEEN, lineage_separated_features, make_window_mech  # noqa: E402
from run_r5b import L, K, AUG_D, H_GRID, ALPHAS, TASKS, episodes, fit  # noqa: E402

SEED = 20261101
BLOCKS = (4, 5, 6, 10, 11, 12)   # A7: blocks 7-9 invalid (IMSIs beyond provisioned range), replaced by 10-12
MECHS = ("iid",) + UNSEEN
METHODS = ["stacking", "augmented", "a4", "a4_incl", "la_aug", "gen_bma", "naive_bayes", "tb_loglinear",
           "pact_declared", "presence", "matched"]


def main():
    units = pd.read_csv(sys.argv[1]).replace([np.inf], 1e9); out_dir = sys.argv[2]
    if "--dryrun" in sys.argv:       # code test only, R4 episodes as six pseudo-blocks; metrics not inspected
        units = units[(units.block.isna()) & (~units.pilot)].copy(); units["block"] = units.episode % 6 + 4
    else:
        units = units[units.block.isin(BLOCKS)].copy()
        bad = units[(units.label == "net_delay") & (units.netem_ok != True)].episode.unique()  # noqa: E712
        units = units[~units.episode.isin(bad)]
        e1 = units.drop_duplicates("episode"); dup = e1[e1.duplicated(["block", "sched_pos"])].episode
        units = units[~units.episode.isin(dup)]
    rows, gate = [], []
    for task in TASKS:
        sub = units[units.label.isin(["healthy", task])].copy()
        if (sub.label == task).sum() == 0:
            continue
        sub["y"] = (sub.label == task).astype(int)
        for held in BLOCKS:
            pool, te = sub[sub.block != held], sub[sub.block == held]
            if te.y.nunique() < 2:
                continue
            for seed in range(3):
                rng = np.random.default_rng(SEED + 100 * held + seed + 7 * TASKS.index(task))
                eps = pool.drop_duplicates("episode")[["episode", "y"]]; tr_e, ca_e = [], []
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
                Wa, ya = [], []
                for rep in range(5):
                    for _, y, z in E_tr:
                        Wa.append(make_window(z, y, int(r1.choice(AUG_D)), float(r1.choice(H_GRID)), r1)); ya.append(y)
                au = fit([L.count_features(w) for w in Wa], ya)
                la = fit([lineage_separated_features(L, w) for w in Wa], ya)
                Wca = [make_window(z, y, 1, 0.0, r1) for _, y, z in E_ca]
                auc_cal = M.auroc(st.predict_proba(np.array([L.count_features(w) for w in Wca]))[:, 1], yca)
                gate.append(dict(task=task, held_block=held, seed=seed, auc_cal=auc_cal, passed=bool(auc_cal >= 0.65)))
                Z = np.array([zz for _, _, z in E_tr for zz in z.values()]); H = np.array([y for _, y, z in E_tr for _ in z])
                m = fit_origin(Z, H); prior = float(np.mean(ytr))
                alpha = min(ALPHAS, key=lambda a: M.brier([loglinear_pool(w, {o: m for o in w.z}, prior, a) for w in Wca], yca))
                for mech in MECHS:
                    for h in H_GRID:
                        r3 = np.random.default_rng(SEED * 13 + seed * 101 + MECHS.index(mech) * 17 + int(h * 100) + held * 1000 + TASKS.index(task) * 7)
                        ma = fit([L.count_features(make_window_mech(z, y, mech, h, r3)) for _, y, z in E_tr for _ in range(3)],
                                 [y for _, y, _ in E_tr for _ in range(3)])
                        P = {k: [] for k in METHODS}; Lo, Up = [], []
                        for _, y, z in E_te:
                            w = make_window_mech(z, y, mech, h, r3); models = {o: m for o in w.z}
                            c = L.count_features(w)[None]
                            P["stacking"].append(st.predict_proba(c)[0, 1])
                            P["augmented"].append(au.predict_proba(c)[0, 1])
                            p, lo, up = L.lineage_stacking(w, a4); P["a4"].append(p); Lo.append(lo); Up.append(up)
                            P["a4_incl"].append(a4.predict_proba(L.unit_features(w, "include")[None])[0, 1])
                            P["la_aug"].append(la.predict_proba(lineage_separated_features(L, w)[None])[0, 1])
                            g = fuse_generative(w, models, prior)
                            P["gen_bma"].append(g["paef_bma"]); P["naive_bayes"].append(g["naive_bayes"])
                            P["tb_loglinear"].append(loglinear_pool(w, models, prior, alpha))
                            P["pact_declared"].append(pact_meet(w, models, prior, "declared"))
                            P["presence"].append(pr.predict_proba(L.presence_features(w)[None])[0, 1])
                            P["matched"].append(ma.predict_proba(c)[0, 1])
                        Lo, Up = np.array(Lo), np.array(Up); dec = (Lo > .5) | (Up < .5)
                        rec = dict(task=task, held_block=held, seed=seed, mech=mech, h=h, n_test=len(yte), alpha=float(alpha),
                                   dec_share=float(dec.mean()),
                                   dec_err=float(np.mean((Lo[dec] > .5).astype(int) != yte[dec])) if dec.any() else np.nan,
                                   a4_err=float(np.mean((np.array(P["a4"]) > .5).astype(int) != yte)))
                        for k, p in P.items():
                            p = np.array(p)
                            rec[f"{k}_brier"] = M.brier(p, yte); rec[f"{k}_nll"] = M.nll(p, yte)
                            rec[f"{k}_ece"] = M.ece(p, yte); rec[f"{k}_auc"] = M.auroc(p, yte)
                        rows.append(rec)
            print(task, "held", held, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "r6_cells.csv"), index=False)
    pd.DataFrame(gate).to_csv(os.path.join(out_dir, "r6_gate.csv"), index=False)


if __name__ == "__main__":
    main()
