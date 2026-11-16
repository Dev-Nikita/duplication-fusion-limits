"""R10 analysis (docs/r10-protocol.md). Usage: python3 -I run_r10.py <units.csv> <out_dir> [--dryrun]"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "r5b"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import make_window  # noqa: E402
from paef.real.mechanisms import lineage_separated_features  # noqa: E402
from paef.real.mechanisms_r10 import LI, LD_PAIR, make_window_r10  # noqa: E402
from run_r5b import L, K, AUG_D, H_GRID, episodes, fit  # noqa: E402

SEED = 20261101 + 13
BLOCKS = (13, 14, 15, 16, 17, 18)
TASKS = ("amf_stall", "ausf_stall", "net_delay", "reg_reject")       # reg_reject reported separately
GB_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=n) for d in (2, 3) for lr in (0.05, 0.1) for n in (100, 300)]
ERR_EPS, STORM_SIZES = 0.2, (4, 16)


def gb_fit(X, y, Xc, yc):
    best = min(GB_GRID, key=lambda p: M.brier(HistGradientBoostingClassifier(random_state=0, **p).fit(X, y).predict_proba(Xc)[:, 1], yc))
    return HistGradientBoostingClassifier(random_state=0, **best).fit(X, y)


def unit_feats(w):                      # exact lineage: units only, every copy removed
    return L.unit_features(w)


def err_feats(w, kind, eps, rng):
    f = lineage_separated_features(L, w).copy(); units = f[:2 * K].reshape(K, 2); rec = f[2 * K:3 * K]
    if kind == "split":
        for _, k, hid in w.copies:
            if not hid and rng.random() < eps:
                rec[k] -= 1; units[k, 1] += 1
    else:
        for zz in w.z.values():
            for k, v in enumerate(zz):
                if v == 1 and rng.random() < eps:
                    units[k, 1] -= 1; rec[k] += 1
    f[:2 * K] = units.ravel(); f[2 * K:3 * K] = rec
    return f


def main():
    u = pd.read_csv(sys.argv[1]).replace([np.inf], 1e9); out = sys.argv[2]
    blocks = BLOCKS
    if "--dryrun" in sys.argv:          # code test on R6 blocks; metrics not inspected
        u = u[u.block.isin([4, 5, 6, 10, 11, 12])].copy(); blocks = (4, 5, 6, 10, 11, 12)
    else:
        u = u[u.block.isin(BLOCKS)].copy()
        bad = u[(u.label == "net_delay") & (u.netem_ok != True)].episode.unique(); u = u[~u.episode.isin(bad)]  # noqa: E712
    rows, gate = [], []
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for held in blocks:
            pool, te = sub[sub.block != held], sub[sub.block == held]
            if te.y.nunique() < 2:
                continue
            for seed in range(3):
                rng = np.random.default_rng(SEED + 100 * held + seed + 7 * TASKS.index(task))
                eps_ = pool.drop_duplicates("episode")[["episode", "y"]]; tr_e, ca_e = [], []
                for _, g in eps_.groupby("y"):
                    idx = rng.permutation(len(g)); a = int(round(0.7 * len(g)))
                    tr_e += list(g.episode.iloc[idx[:a]]); ca_e += list(g.episode.iloc[idx[a:]])
                tr, ca = pool[pool.episode.isin(tr_e)], pool[pool.episode.isin(ca_e)]; h0 = tr[tr.y == 0]
                thr = (float(np.percentile(h0.ue_lat, 90)), float(np.percentile(h0.amf_lat, 90)))
                E_tr, E_ca, E_te = episodes(tr, thr), episodes(ca, thr), episodes(te, thr)
                yca = np.array([y for _, y, _ in E_ca]); yte = np.array([y for _, y, _ in E_te])
                r1 = np.random.default_rng(SEED * 3 + seed + held)
                W1 = [make_window(z, y, 1, 0.0, r1) for _, y, z in E_tr]; Wc1 = [make_window(z, y, 1, 0.0, r1) for _, y, z in E_ca]
                st = fit([L.count_features(w) for w in W1], [w.y for w in W1])
                gate.append(dict(task=task, held_block=held, seed=seed,
                                 auc_cal=M.auroc(st.predict_proba(np.array([L.count_features(w) for w in Wc1]))[:, 1], yca)))
                Wa, Wc = [], []
                for rep in range(5):
                    Wa += [make_window(z, y, int(r1.choice(AUG_D)), float(r1.choice(H_GRID)), r1) for _, y, z in E_tr]
                for rep in range(3):
                    Wc += [make_window(z, y, int(r1.choice(AUG_D)), float(r1.choice(H_GRID)), r1) for _, y, z in E_ca]
                ya, yc = [w.y for w in Wa], np.array([w.y for w in Wc])
                Ca, Cc = np.array([L.count_features(w) for w in Wa]), np.array([L.count_features(w) for w in Wc])
                Fa, Fc = np.array([lineage_separated_features(L, w) for w in Wa]), np.array([lineage_separated_features(L, w) for w in Wc])
                Ua, Uc = np.array([unit_feats(w) for w in Wa]), np.array([unit_feats(w) for w in Wc])
                aug = fit(Ca, ya); la = fit(Fa, ya)
                gb_aug = gb_fit(Ca, ya, Cc, yc); gb_la = gb_fit(Fa, ya, Fc, yc); gb_unit = gb_fit(Ua, ya, Uc, yc)
                r3 = np.random.default_rng(SEED * 13 + seed * 101 + held * 1000 + TASKS.index(task) * 7)
                # mixture oracle for the LD pair: GBDT on counts trained on labelled windows of BOTH deployment mechanisms
                Wmix = [make_window_r10(z, y, m, 0.5, r3) for m in LD_PAIR for _, y, z in E_tr for _ in range(3)]
                Wmixc = [make_window_r10(z, y, m, 0.5, r3) for m in LD_PAIR for _, y, z in E_ca for _ in range(2)]
                gb_mix = gb_fit(np.array([L.count_features(w) for w in Wmix]), [w.y for w in Wmix],
                                np.array([L.count_features(w) for w in Wmixc]), np.array([w.y for w in Wmixc]))
                for mech in LI + LD_PAIR:
                    W = [make_window_r10(z, y, mech, 0.5, r3) for _, y, z in E_te]
                    C = np.array([L.count_features(w) for w in W]); F = np.array([lineage_separated_features(L, w) for w in W])
                    U = np.array([unit_feats(w) for w in W])
                    rec = dict(task=task, held_block=held, seed=seed, mech=mech, arm="LI" if mech in LI else "LD", n=len(yte),
                               augmented=M.brier(aug.predict_proba(C)[:, 1], yte), la_aug=M.brier(la.predict_proba(F)[:, 1], yte),
                               gbdt_aug=M.brier(gb_aug.predict_proba(C)[:, 1], yte), gbdt_la=M.brier(gb_la.predict_proba(F)[:, 1], yte),
                               gbdt_unit=M.brier(gb_unit.predict_proba(U)[:, 1], yte),
                               gbdt_mix_oracle=M.brier(gb_mix.predict_proba(C)[:, 1], yte))
                    rows.append(rec)
                for size in STORM_SIZES:      # ERR arm: label-independent storms of two sizes
                    W = [make_window_r10(z, y, "li_storm", 0.0, r3, storm_size=size) for _, y, z in E_te]
                    clean = M.brier(la.predict_proba(np.array([lineage_separated_features(L, w) for w in W]))[:, 1], yte)
                    for kind in ("split", "merge"):
                        Fe = np.array([err_feats(w, kind, ERR_EPS, r3) for w in W])
                        rows.append(dict(task=task, held_block=held, seed=seed, mech=f"err_{kind}_{size}", arm="ERR", n=len(yte),
                                         harm=M.brier(la.predict_proba(Fe)[:, 1], yte) - clean))
            print(task, held, flush=True)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "r10_cells.csv"), index=False)
    pd.DataFrame(gate).to_csv(os.path.join(out, "r10_gate.csv"), index=False)


if __name__ == "__main__":
    main()
