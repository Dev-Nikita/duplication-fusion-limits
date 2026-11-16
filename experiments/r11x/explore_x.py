"""R11-X exploratory arm (X4-X7; docs/r11x-protocol.md, addendum A). NOT confirmatory.
Same episodes, thresholds, i.i.d. multiplicity augmentation, duplication mechanisms and leave-one-block-out
as the frozen R11 pipeline. One reference for every channel: gbdt_unit = GBDT on the number of distinct
firing units per reading type (= exact lineage, i.e. every copy removed).
Observation channels (features available to the lineage-free learners):
  A      firing-message counts per reading type                                   (X1 channel)
  A+U    A + oracle number of distinct firing units per reading type              (X4)
  A+M    A + number of reporting units in the window, no negative reports         (X5)
  F      A + non-firing report counts (the denominator)                            (X3 channel)
  F_q30  F with each non-firing report lost independently with prob. 0.3          (X6, negative-report loss)
  Fv     F on windows whose reporting population is random, M ~ U{8..20}          (X7, unknown population)
  LA     lineage-separated firing features (units, recorded, unrecorded copies)    (lineage-aware check)
Learners: GBDT (same grid as R10/R11) and logistic regression. Metrics: Brier and log loss.
Usage: python3 -I explore_x.py <units.csv> <out_dir> <comma-separated blocks>"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../../src", "../r5b", "../r10"):
    sys.path.insert(0, os.path.join(HERE, p))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import make_window  # noqa: E402
from paef.real.mechanisms import lineage_separated_features  # noqa: E402
from paef.real.mechanisms_r10 import LI, LD_PAIR, make_window_r10  # noqa: E402
from run_r5b import L, AUG_D, H_GRID, episodes, fit  # noqa: E402
from run_r10 import gb_fit  # noqa: E402

K = L.K; SEED = 20261024; TASKS = ("amf_stall", "ausf_stall", "net_delay"); MECHS = ("none",) + LI + LD_PAIR


def counts(w):
    return L.count_features(w).reshape(K, 2)          # [:,0] non-firing reports, [:,1] firing messages


def units(w):
    return L.unit_features(w).reshape(K, 2)[:, 1]     # distinct firing units


def feats(w, ch, rng):
    c = counts(w)
    if ch == "A":
        return c[:, 1].copy()
    if ch == "A+U":
        return np.concatenate([c[:, 1], units(w)])
    if ch == "A+M":
        return np.concatenate([c[:, 1], [len(w.z)]])
    if ch in ("F", "Fv"):
        return c.ravel().copy()
    if ch == "F_q30":
        neg = rng.binomial(c[:, 0].astype(int), 0.7)
        return np.concatenate([neg, c[:, 1]])
    if ch == "LA":
        f = lineage_separated_features(L, w); return np.concatenate([f[:2 * K].reshape(K, 2)[:, 1], f[2 * K:]])
    raise ValueError(ch)


CH_STD = ("A", "A+U", "A+M", "F", "F_q30", "LA")


def subsample(E, rng):
    out = []
    for ep, y, z in E:
        keys = list(z); m = int(rng.integers(8, min(20, len(keys)) + 1))
        keep = rng.choice(len(keys), m, replace=False)
        out.append((ep, y, {keys[i]: z[keys[i]] for i in keep}))
    return out


def run_variant(E_tr, E_ca, E_te, yte, chans, rng, tag, rows, meta):
    Wa, Wc = [], []
    for _ in range(5):
        Wa += [make_window(z, y, int(rng.choice(AUG_D)), float(rng.choice(H_GRID)), rng) for _, y, z in E_tr]
    for _ in range(3):
        Wc += [make_window(z, y, int(rng.choice(AUG_D)), float(rng.choice(H_GRID)), rng) for _, y, z in E_ca]
    ya, yc = np.array([w.y for w in Wa]), np.array([w.y for w in Wc])
    Ua, Uc = np.array([units(w) for w in Wa]), np.array([units(w) for w in Wc])
    models = {"gbdt_unit": ("U", gb_fit(Ua, ya, Uc, yc))}
    for ch in chans:
        Xa = np.array([feats(w, ch, rng) for w in Wa]); Xc = np.array([feats(w, ch, rng) for w in Wc])
        models[f"gbdt|{ch}"] = (ch, gb_fit(Xa, ya, Xc, yc))
        if ch != "LA":
            models[f"logit|{ch}"] = (ch, fit(Xa, ya))
    for mech in MECHS:
        W = [make_window_r10(z, y, mech, 0.5, rng) for _, y, z in E_te]
        X = {"U": np.array([units(w) for w in W])}
        for ch in chans:
            X[ch] = np.array([feats(w, ch, rng) for w in W])
        for name, (ch, mdl) in models.items():
            p = mdl.predict_proba(X[ch])[:, 1]
            rows.append(dict(**meta, variant=tag, mech=mech, method=name, brier=M.brier(p, yte), nll=M.nll(p, yte)))


def main():
    u = pd.read_csv(sys.argv[1]).replace([np.inf], 1e9); out = sys.argv[2]
    blocks = tuple(int(b) for b in sys.argv[3].split(","))
    u = u[u.block.isin(blocks)].copy()
    bad = u[(u.label == "net_delay") & (u.netem_ok != True)].episode.unique(); u = u[~u.episode.isin(bad)]  # noqa: E712
    rows = []
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for held in blocks:
            pool, te = sub[sub.block != held], sub[sub.block == held]
            if te.y.nunique() < 2:
                continue
            for seed in range(3):
                rng = np.random.default_rng(SEED + 1000 * held + 10 * seed + TASKS.index(task))
                eps_ = pool.drop_duplicates("episode")[["episode", "y"]]; tr_e, ca_e = [], []
                for _, g in eps_.groupby("y"):
                    idx = rng.permutation(len(g)); a = int(round(0.7 * len(g)))
                    tr_e += list(g.episode.iloc[idx[:a]]); ca_e += list(g.episode.iloc[idx[a:]])
                tr, ca = pool[pool.episode.isin(tr_e)], pool[pool.episode.isin(ca_e)]; h0 = tr[tr.y == 0]
                thr = (float(np.percentile(h0.ue_lat, 90)), float(np.percentile(h0.amf_lat, 90)))
                E_tr, E_ca, E_te = episodes(tr, thr), episodes(ca, thr), episodes(te, thr)
                yte = np.array([y for _, y, _ in E_te]); meta = dict(task=task, held_block=held, seed=seed, n=len(yte))
                run_variant(E_tr, E_ca, E_te, yte, CH_STD, rng, "fixedM", rows, meta)
                run_variant(subsample(E_tr, rng), subsample(E_ca, rng), subsample(E_te, rng), yte, ("A", "F", "LA"), rng, "varM", rows, meta)
            print(task, held, flush=True)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "explore_cells.csv"), index=False)


if __name__ == "__main__":
    main()
