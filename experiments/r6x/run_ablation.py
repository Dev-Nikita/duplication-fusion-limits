"""R6-X ablation (docs/r6x-protocol.md): is the la_aug gain due to provenance or to extra capacity?
Reproduces the R6 training/test windows exactly (same RNG streams as run_r6.py) and adds:
  la_placebo  - same feature layout as la_aug, but the copy labels (recorded / unrecorded) are assigned to
                randomly chosen messages of the window, preserving how many of each there are;
  aug_cap     - augmented stacking with the same number of features as la_aug (counts + log1p + sqrt of
                firing counts), no provenance;
  aug_tuned / la_tuned - both with the L2 strength C chosen on augmented calibration windows.
Usage: python3 -I run_ablation.py <r6 units.csv> <out_dir>"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "r5b")); sys.path.insert(0, os.path.join(HERE, "..", "r6"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import make_window  # noqa: E402
from paef.real.mechanisms import lineage_separated_features, make_window_mech  # noqa: E402
from run_r5b import L, K, AUG_D, H_GRID, TASKS, episodes, fit  # noqa: E402
from run_r6 import SEED, BLOCKS, MECHS  # noqa: E402

CS = (0.01, 0.1, 1.0, 10.0)


def placebo_features(w, rng):
    msgs = [(k, int(v)) for zz in w.z.values() for k, v in enumerate(zz)] + [(k, 1) for _, k, _ in w.copies]
    n_rec = sum(1 for c in w.copies if not c[2]); n_unr = len(w.copies) - n_rec
    idx = rng.permutation(len(msgs)); rec, unr = set(idx[:n_rec]), set(idx[n_rec:n_rec + n_unr])
    units = np.zeros((K, 2)); kr = np.zeros(K); ku = np.zeros(K)
    for i, (k, v) in enumerate(msgs):
        if i in rec:
            kr[k] += 1
        elif i in unr:
            ku[k] += 1
        else:
            units[k, v] += 1
    return np.concatenate([units.ravel(), kr, ku])


def cap_features(w):
    c = L.count_features(w); fire = c.reshape(K, 2)[:, 1]
    return np.concatenate([c, np.log1p(fire), np.sqrt(fire)])


def fitC(X, y, C):
    return LogisticRegression(C=C, max_iter=5000).fit(np.array(X), np.array(y))


def main():
    u = pd.read_csv(sys.argv[1]).replace([np.inf], 1e9); out = sys.argv[2]
    u = u[u.block.isin(BLOCKS)].copy()
    rows = []
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for held in BLOCKS:
            pool, te = sub[sub.block != held], sub[sub.block == held]
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
                ytr = [y for _, y, _ in E_tr]; yte = np.array([y for _, y, _ in E_te])
                r1 = np.random.default_rng(SEED * 3 + seed + held)
                [make_window(z, y, 1, 0.0, r1) for _, y, z in E_tr]               # consume as in run_r6
                Wa, ya = [], []
                for rep in range(5):
                    for _, y, z in E_tr:
                        Wa.append(make_window(z, y, int(r1.choice(AUG_D)), float(r1.choice(H_GRID)), r1)); ya.append(y)
                rp = np.random.default_rng(SEED * 7 + seed + held * 31 + TASKS.index(task))   # placebo / tuning stream
                au = fit([L.count_features(w) for w in Wa], ya)
                la = fit([lineage_separated_features(L, w) for w in Wa], ya)
                pl = fit([placebo_features(w, rp) for w in Wa], ya)
                cp = fit([cap_features(w) for w in Wa], ya)
                Wc = [make_window(z, y, int(rp.choice(AUG_D)), float(rp.choice(H_GRID)), rp) for _ in range(3) for _, y, z in E_ca]
                yc = np.array([w.y for w in Wc])
                def tune(feat):
                    Xtr, Xc = [feat(w) for w in Wa], np.array([feat(w) for w in Wc])
                    best = min(CS, key=lambda C: M.brier(fitC(Xtr, ya, C).predict_proba(Xc)[:, 1], yc))
                    return fitC(Xtr, ya, best), best
                aut, cA = tune(L.count_features); lat, cL = tune(lambda w: lineage_separated_features(L, w))
                for mech in MECHS:
                    h = 0.5
                    r3 = np.random.default_rng(SEED * 13 + seed * 101 + MECHS.index(mech) * 17 + int(h * 100) + held * 1000 + TASKS.index(task) * 7)
                    [make_window_mech(z, y, mech, h, r3) for _, y, z in E_tr for _ in range(3)]   # consume as matched training
                    W = [make_window_mech(z, y, mech, h, r3) for _, y, z in E_te]
                    P = {"augmented": au.predict_proba(np.array([L.count_features(w) for w in W]))[:, 1],
                         "la_aug": la.predict_proba(np.array([lineage_separated_features(L, w) for w in W]))[:, 1],
                         "la_placebo": pl.predict_proba(np.array([placebo_features(w, rp) for w in W]))[:, 1],
                         "aug_cap": cp.predict_proba(np.array([cap_features(w) for w in W]))[:, 1],
                         "aug_tuned": aut.predict_proba(np.array([L.count_features(w) for w in W]))[:, 1],
                         "la_tuned": lat.predict_proba(np.array([lineage_separated_features(L, w) for w in W]))[:, 1]}
                    rec = dict(task=task, held_block=held, seed=seed, mech=mech, h=h, C_aug=cA, C_la=cL)
                    for k, p in P.items():
                        rec[f"{k}_brier"] = M.brier(p, yte); rec[f"{k}_nll"] = M.nll(p, yte)
                    rows.append(rec)
            print(task, held, flush=True)
    os.makedirs(out, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(out, "r6x_cells.csv"), index=False)


if __name__ == "__main__":
    main()
