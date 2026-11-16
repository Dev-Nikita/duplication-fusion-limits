"""EXPLORATORY (post-hoc on R5 data, 2026-10-18, after B2 failed): lineage-aware augmented stacking.
Features = recorded units (known copies dropped) + count of copies with unrecorded lineage per slot,
trained with the same duplication augmentation as the 'augmented' baseline. Not confirmatory."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../../src"))
import run_r5b as R  # noqa: E402
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import make_window  # noqa: E402
L = R.L


def la_feat(w):
    hid = np.zeros(R.K)
    for _, k, hd in w.copies:
        if hd:
            hid[k] += 1
    return np.concatenate([L.unit_features(w), hid])


units = pd.read_csv(sys.argv[1]).replace([np.inf], 1e9); units = units[units.block.isin([1, 2, 3])]
rows = []
for task in R.TASKS:
    sub = units[units.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
    for held in (1, 2, 3):
        pool, te = sub[sub.block != held], sub[sub.block == held]
        for seed in range(5):
            rng = np.random.default_rng(R.SEED + 100 * held + seed + 7 * R.TASKS.index(task))
            eps = pool.drop_duplicates("episode")[["episode", "y"]]; tr_e = []
            for _, g in eps.groupby("y"):
                idx = rng.permutation(len(g)); tr_e += list(g.episode.iloc[idx[:int(round(0.7 * len(g)))]])
            tr = pool[pool.episode.isin(tr_e)]; h0 = tr[tr.y == 0]
            thr = (float(np.percentile(h0.ue_lat, 90)), float(np.percentile(h0.amf_lat, 90)))
            E_tr, E_te = R.episodes(tr, thr), R.episodes(te, thr); ytr = [y for _, y, _ in E_tr]; yte = np.array([y for _, y, _ in E_te])
            r1 = np.random.default_rng(seed + 99 * held)
            Wa, ya = [], []
            for rep in range(5):
                for _, y, z in E_tr:
                    Wa.append(make_window(z, y, int(r1.choice(R.AUG_D)), float(r1.choice(R.H_GRID)), r1)); ya.append(y)
            au = R.fit([L.count_features(w) for w in Wa], ya); la = R.fit([la_feat(w) for w in Wa], ya)
            for D, h in [(1, .5), (8, .5), (16, .5), (8, 1.0), (16, 1.0)]:
                r3 = np.random.default_rng(seed * 7 + D + int(h * 10) + held * 1000)
                W = [make_window(z, y, D, h, r3) for _, y, z in E_te]
                pa = au.predict_proba(np.array([L.count_features(w) for w in W]))[:, 1]
                pl = la.predict_proba(np.array([la_feat(w) for w in W]))[:, 1]
                rows.append(dict(task=task, held=held, seed=seed, D=D, h=h, augmented=M.brier(pa, yte), lineage_aug=M.brier(pl, yte)))
d = pd.DataFrame(rows); d.to_csv(os.path.join(os.path.dirname(sys.argv[1]), "explore_lineage_aug.csv"), index=False)
c = d.groupby(["D", "h", "task", "held"])[["augmented", "lineage_aug"]].mean().reset_index()
c["diff"] = c.augmented - c.lineage_aug
print(c.groupby(["D", "h"])[["augmented", "lineage_aug", "diff"]].mean().round(4))
print(c.groupby(["D", "h", "held"])["diff"].mean().unstack().round(4))
