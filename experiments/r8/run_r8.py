"""R8: documented Alertmanager behaviour applied to the R6 episodes (docs/r8-protocol.md). Training is identical
to R6 (same splits, same i.i.d. augmentation); only the test-time pipeline changes.
Usage: python3 -I run_r8.py <r6 units.csv> <out_dir>"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "r5b")); sys.path.insert(0, os.path.join(HERE, "..", "r6"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import make_window  # noqa: E402
from paef.real.mechanisms import lineage_separated_features  # noqa: E402
from paef.real.alertmanager import SCENARIOS, make_window_am, count_features_am, unit_features_am, la_features_am, a4_am  # noqa: E402
from run_r5b import L, AUG_D, H_GRID, TASKS, episodes, fit  # noqa: E402
from run_r6 import SEED, BLOCKS  # noqa: E402

METHODS = ["stacking", "augmented", "a4", "la_aug", "presence", "matched"]


def main():
    u = pd.read_csv(sys.argv[1]).replace([np.inf], 1e9); out = sys.argv[2]
    u = u[u.block.isin(BLOCKS)].copy(); rows = []
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for held in BLOCKS:
            pool, te = sub[sub.block != held], sub[sub.block == held]
            for seed in range(3):
                rng = np.random.default_rng(SEED + 100 * held + seed + 7 * TASKS.index(task))
                eps = pool.drop_duplicates("episode")[["episode", "y"]]; tr_e = []
                for _, g in eps.groupby("y"):
                    idx = rng.permutation(len(g)); tr_e += list(g.episode.iloc[idx[:int(round(0.7 * len(g)))]])
                tr = pool[pool.episode.isin(tr_e)]; h0 = tr[tr.y == 0]
                thr = (float(np.percentile(h0.ue_lat, 90)), float(np.percentile(h0.amf_lat, 90)))
                E_tr, E_te = episodes(tr, thr), episodes(te, thr)
                ytr = [y for _, y, _ in E_tr]; yte = np.array([y for _, y, _ in E_te])
                r1 = np.random.default_rng(SEED * 3 + seed + held)
                W1 = [make_window(z, y, 1, 0.0, r1) for _, y, z in E_tr]
                st = fit([L.count_features(w) for w in W1], ytr); a4 = fit([L.unit_features(w) for w in W1], ytr)
                pr = fit([L.presence_features(w) for w in W1], ytr)
                Wa, ya = [], []
                for rep in range(5):
                    for _, y, z in E_tr:
                        Wa.append(make_window(z, y, int(r1.choice(AUG_D)), float(r1.choice(H_GRID)), r1)); ya.append(y)
                au = fit([L.count_features(w) for w in Wa], ya); la = fit([lineage_separated_features(L, w) for w in Wa], ya)
                for sc in SCENARIOS:
                    r3 = np.random.default_rng(SEED * 17 + seed * 101 + SCENARIOS.index(sc) * 13 + held * 1000 + TASKS.index(task) * 7)
                    ma = fit([count_features_am(L, make_window_am(z, y, sc, r3)) for _, y, z in E_tr], ytr)
                    W = [make_window_am(z, y, sc, r3) for _, y, z in E_te]
                    C = np.array([count_features_am(L, w) for w in W])
                    P = {"stacking": st.predict_proba(C)[:, 1], "augmented": au.predict_proba(C)[:, 1],
                         "a4": np.array([a4_am(L, w, a4)[0] for w in W]),
                         "la_aug": la.predict_proba(np.array([la_features_am(L, w) for w in W]))[:, 1],
                         "presence": pr.predict_proba((C.reshape(len(W), -1, 2)[:, :, 1] > 0).astype(float))[:, 1],
                         "matched": ma.predict_proba(C)[:, 1]}
                    rec = dict(task=task, held_block=held, seed=seed, scenario=sc, n_copies=float(np.mean([len(w.copies) for w in W])),
                               n_invisible=float(np.mean([len(w.invisible) for w in W])))
                    for k, p in P.items():
                        rec[f"{k}_brier"] = M.brier(p, yte); rec[f"{k}_nll"] = M.nll(p, yte)
                    rows.append(rec)
            print(task, held, flush=True)
    os.makedirs(out, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(out, "r8_cells.csv"), index=False)


if __name__ == "__main__":
    main()
