"""R8 re-analysis on two observation channels (EXPLORATORY, after R11-X; docs/r11x-protocol.md addendum A).
Same R6 episodes, split, thresholds and i.i.d. multiplicity augmentation as R8; the Alertmanager emulation
(src/paef/real/alertmanager.py, frozen for R8) is unchanged. Two channels are compared:
  full  : firing-message counts + non-firing report counts per reading type (as in R8)
  alarm : firing-message counts only (what an Alertmanager receiver actually gets)
Lineage-aware references: distinct firing units (HA-replica duplicates have distinct fingerprints and are
therefore counted as units, as in R8) and lineage-separated features.
Usage: python3 -I explore_r8.py <r6 units.csv> <out_dir>"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../../src", "../r5b", "../r6", "../r10"):
    sys.path.insert(0, os.path.join(HERE, p))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import make_window  # noqa: E402
from paef.real.alertmanager import SCENARIOS, make_window_am  # noqa: E402
from run_r5b import L, AUG_D, H_GRID, episodes, fit  # noqa: E402
from run_r6 import SEED, BLOCKS  # noqa: E402
from run_r10 import gb_fit  # noqa: E402

K = L.K; TASKS = ("amf_stall", "ausf_stall", "net_delay")


def base(w):
    f = np.zeros((K, 2)); u = np.zeros(K); rec = np.zeros(K); hid = np.zeros(K)
    for zz in w.z.values():
        for k, v in enumerate(zz):
            f[k, v] += 1
    u[:] = f[:, 1]
    for _, k in getattr(w, "invisible", []):
        f[k, 1] += 1; u[k] += 1
    for _, k, hd in w.copies:
        f[k, 1] += 1; (hid if hd else rec)[k] += 1
    return f, u, rec, hid


def feats(w, kind):
    f, u, rec, hid = base(w)
    return {"full": f.ravel(), "alarm": f[:, 1].copy(), "unit": u, "la": np.concatenate([u, rec, hid])}[kind]


def main():
    uu = pd.read_csv(sys.argv[1]).replace([np.inf], 1e9); out = sys.argv[2]
    uu = uu[uu.block.isin(BLOCKS)].copy(); rows = []
    for task in TASKS:
        sub = uu[uu.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for held in BLOCKS:
            pool, te = sub[sub.block != held], sub[sub.block == held]
            for seed in range(3):
                rng = np.random.default_rng(SEED + 100 * held + seed + 7 * TASKS.index(task) + 555)
                eps = pool.drop_duplicates("episode")[["episode", "y"]]; tr_e, ca_e = [], []
                for _, g in eps.groupby("y"):
                    idx = rng.permutation(len(g)); a = int(round(0.7 * len(g)))
                    tr_e += list(g.episode.iloc[idx[:a]]); ca_e += list(g.episode.iloc[idx[a:]])
                tr, ca = pool[pool.episode.isin(tr_e)], pool[pool.episode.isin(ca_e)]; h0 = tr[tr.y == 0]
                thr = (float(np.percentile(h0.ue_lat, 90)), float(np.percentile(h0.amf_lat, 90)))
                E_tr, E_ca, E_te = episodes(tr, thr), episodes(ca, thr), episodes(te, thr)
                yte = np.array([y for _, y, _ in E_te])
                Wa = [make_window(z, y, int(rng.choice(AUG_D)), float(rng.choice(H_GRID)), rng) for _ in range(5) for _, y, z in E_tr]
                Wc = [make_window(z, y, int(rng.choice(AUG_D)), float(rng.choice(H_GRID)), rng) for _ in range(3) for _, y, z in E_ca]
                ya, yc = np.array([w.y for w in Wa]), np.array([w.y for w in Wc])
                mdl = {}
                for kind in ("full", "alarm", "unit", "la"):
                    Xa = np.array([feats(w, kind) for w in Wa]); Xc = np.array([feats(w, kind) for w in Wc])
                    mdl[f"gbdt|{kind}"] = (kind, gb_fit(Xa, ya, Xc, yc)); mdl[f"logit|{kind}"] = (kind, fit(Xa, ya))
                for sc in SCENARIOS:
                    W = [make_window_am(z, y, sc, rng) for _, y, z in E_te]
                    X = {k: np.array([feats(w, k) for w in W]) for k in ("full", "alarm", "unit", "la")}
                    for name, (kind, m) in mdl.items():
                        p = m.predict_proba(X[kind])[:, 1]
                        rows.append(dict(task=task, held_block=held, seed=seed, scenario=sc, method=name,
                                         brier=M.brier(p, yte), nll=M.nll(p, yte)))
            print(task, held, flush=True)
    os.makedirs(out, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(out, "r8x_cells.csv"), index=False)


if __name__ == "__main__":
    main()
