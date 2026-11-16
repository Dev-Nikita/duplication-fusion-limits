"""R9 (docs/r9-protocol.md): (a) non-linear lineage-free baseline (gradient-boosted trees, tuned on calibration
windows with the same budget as la_gbdt); (b) lineage ERRORS at test time: false split (a recorded copy is
reported as a new unit), false merge (a true unit is reported as a recorded copy), and lineage missing not at
random. R6 episodes and R6 training windows (same random streams as run_r6.py).
Usage: python3 -I run_r9.py <r6 units.csv> <out_dir>"""
from __future__ import annotations
import os, sys, copy
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "r5b")); sys.path.insert(0, os.path.join(HERE, "..", "r6"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import make_window  # noqa: E402
from paef.real.mechanisms import lineage_separated_features, make_window_mech, UNSEEN  # noqa: E402
from run_r5b import L, K, AUG_D, H_GRID, TASKS, episodes, fit  # noqa: E402
from run_r6 import SEED, BLOCKS  # noqa: E402

ERR = (0.0, 0.05, 0.1, 0.2, 0.4)
GB_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=n) for d in (2, 3) for lr in (0.05, 0.1) for n in (100, 300)]


def corrupt(w, kind, eps, rng):
    """Return features of la_aug after lineage errors. false_split: each recorded copy is, w.p. eps, reported
    as a new unit (it is counted as a unit instead of a recorded copy). false_merge: each firing unit is, w.p. eps,
    reported as a recorded copy of another alert."""
    f = lineage_separated_features(L, w).copy(); units = f[:2 * K].reshape(K, 2); rec = f[2 * K:3 * K]
    if kind == "false_split":
        for _, k, hid in w.copies:
            if not hid and rng.random() < eps:
                rec[k] -= 1; units[k, 1] += 1
    elif kind == "false_merge":
        for zz in w.z.values():
            for k, v in enumerate(zz):
                if v == 1 and rng.random() < eps:
                    units[k, 1] -= 1; rec[k] += 1
    f[:2 * K] = units.ravel(); f[2 * K:3 * K] = rec
    return f


def make_window_mnar(z, y, mech, rng):
    """Lineage missing not at random: unrecorded with probability 0.75 in incident windows, 0.25 otherwise."""
    return make_window_mech(z, y, mech, 0.75 if y == 1 else 0.25, rng)


def gb_fit(X, y, Xc, yc):
    best = min(GB_GRID, key=lambda p: M.brier(HistGradientBoostingClassifier(random_state=0, **p).fit(X, y).predict_proba(Xc)[:, 1], yc))
    return HistGradientBoostingClassifier(random_state=0, **best).fit(X, y)


def main():
    u = pd.read_csv(sys.argv[1]).replace([np.inf], 1e9); out = sys.argv[2]
    u = u[u.block.isin(BLOCKS)].copy(); rows = []
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for held in BLOCKS:
            pool, te = sub[sub.block != held], sub[sub.block == held]
            for seed in range(3):
                rng = np.random.default_rng(SEED + 100 * held + seed + 7 * TASKS.index(task))
                eps_ = pool.drop_duplicates("episode")[["episode", "y"]]; tr_e, ca_e = [], []
                for _, g in eps_.groupby("y"):
                    idx = rng.permutation(len(g)); a = int(round(0.7 * len(g)))
                    tr_e += list(g.episode.iloc[idx[:a]]); ca_e += list(g.episode.iloc[idx[a:]])
                tr, ca = pool[pool.episode.isin(tr_e)], pool[pool.episode.isin(ca_e)]; h0 = tr[tr.y == 0]
                thr = (float(np.percentile(h0.ue_lat, 90)), float(np.percentile(h0.amf_lat, 90)))
                E_tr, E_ca, E_te = episodes(tr, thr), episodes(ca, thr), episodes(te, thr)
                yte = np.array([y for _, y, _ in E_te])
                r1 = np.random.default_rng(SEED * 3 + seed + held)
                [make_window(z, y, 1, 0.0, r1) for _, y, z in E_tr]
                Wa, ya = [], []
                for rep in range(5):
                    for _, y, z in E_tr:
                        Wa.append(make_window(z, y, int(r1.choice(AUG_D)), float(r1.choice(H_GRID)), r1)); ya.append(y)
                au = fit([L.count_features(w) for w in Wa], ya); la = fit([lineage_separated_features(L, w) for w in Wa], ya)
                rq = np.random.default_rng(SEED * 11 + seed + held * 37 + TASKS.index(task))
                Wc = [make_window(z, y, int(rq.choice(AUG_D)), float(rq.choice(H_GRID)), rq) for _ in range(3) for _, y, z in E_ca]
                yc = np.array([w.y for w in Wc])
                Xa = np.array([L.count_features(w) for w in Wa]); Xl = np.array([lineage_separated_features(L, w) for w in Wa])
                gba = gb_fit(Xa, ya, np.array([L.count_features(w) for w in Wc]), yc)
                gbl = gb_fit(Xl, ya, np.array([lineage_separated_features(L, w) for w in Wc]), yc)
                for mech in UNSEEN:
                    r3 = np.random.default_rng(SEED * 19 + seed * 101 + UNSEEN.index(mech) * 17 + held * 1000 + TASKS.index(task) * 7)
                    W = [make_window_mech(z, y, mech, 0.5, r3) for _, y, z in E_te]
                    C = np.array([L.count_features(w) for w in W]); F = np.array([lineage_separated_features(L, w) for w in W])
                    base = dict(task=task, held_block=held, seed=seed, mech=mech)
                    pa = au.predict_proba(C)[:, 1]
                    rows.append(dict(base, part="gbdt", augmented=M.brier(pa, yte), la_aug=M.brier(la.predict_proba(F)[:, 1], yte),
                                     gbdt_aug=M.brier(gba.predict_proba(C)[:, 1], yte), gbdt_la=M.brier(gbl.predict_proba(F)[:, 1], yte)))
                    for kind in ("false_split", "false_merge"):
                        for e in ERR:
                            Fe = np.array([corrupt(w, kind, e, r3) for w in W])
                            rows.append(dict(base, part=kind, eps=e, augmented=M.brier(pa, yte), la_aug=M.brier(la.predict_proba(Fe)[:, 1], yte)))
                    Wm = [make_window_mnar(z, y, mech, r3) for _, y, z in E_te]
                    rows.append(dict(base, part="mnar", augmented=M.brier(au.predict_proba(np.array([L.count_features(w) for w in Wm]))[:, 1], yte),
                                     la_aug=M.brier(la.predict_proba(np.array([lineage_separated_features(L, w) for w in Wm]))[:, 1], yte)))
            print(task, held, flush=True)
    os.makedirs(out, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(out, "r9_cells.csv"), index=False)


if __name__ == "__main__":
    main()
