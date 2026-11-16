"""R11 analysis (docs/r11-protocol.md). Usage:
  python3 -I add_cond.py <episodes_root> <units.csv> <units_cond.csv>     (adds load_cond from meta.json)
  python3 -I run_r11.py <units_cond.csv> <out_dir> [--dryrun]
Within condition, leave-one-block-out over blocks 21..32. The R10 pipeline is reused unchanged; the
additions are the condition split, 5 seeds, a no-duplication mechanism and a deduplication baseline
(GBDT on presence features). --dryrun: code test on R6 blocks with a random fake condition; metrics are not
inspected."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "r5b"))
sys.path.insert(0, os.path.join(HERE, "..", "r10"))
from paef.evaluation import metrics as M  # noqa: E402
from paef.real.generic import make_window  # noqa: E402
from paef.real.mechanisms import lineage_separated_features  # noqa: E402
from paef.real.mechanisms_r10 import LI, LD_PAIR, make_window_r10  # noqa: E402
from run_r5b import L, AUG_D, H_GRID, episodes, fit  # noqa: E402
from run_r10 import gb_fit, unit_feats  # noqa: E402

SEED = 20261201
BLOCKS = tuple(range(21, 33))
TASKS = ("amf_stall", "ausf_stall", "net_delay")
MECHS = ("none",) + LI + LD_PAIR
N_SEEDS = 5


def main():
    u = pd.read_csv(sys.argv[1]).replace([np.inf], 1e9); out = sys.argv[2]
    blocks = BLOCKS
    if "--dryrun" in sys.argv:
        blocks = (4, 5, 6, 10, 11, 12); u = u[u.block.isin(blocks)].copy()
        rng0 = np.random.default_rng(0); ep = u.episode.unique()
        fake = dict(zip(ep, rng0.choice(["control", "loaded"], len(ep)))); u["load_cond"] = u.episode.map(fake)
    else:
        u = u[u.block.isin(BLOCKS)].copy()
        bad = u[(u.label == "net_delay") & (u.netem_ok != True)].episode.unique(); u = u[~u.episode.isin(bad)]  # noqa: E712
    rows, gate = [], []
    for cond in ("control", "loaded"):
        uc = u[u.load_cond == cond]
        for task in TASKS:
            sub = uc[uc.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
            for held in blocks:
                pool, te = sub[sub.block != held], sub[sub.block == held]
                if te.y.nunique() < 2:
                    continue
                for seed in range(N_SEEDS):
                    k = (0 if cond == "control" else 1, TASKS.index(task))
                    rng = np.random.default_rng(SEED + 100 * held + seed + 7 * k[1] + 10007 * k[0])
                    eps_ = pool.drop_duplicates("episode")[["episode", "y"]]; tr_e, ca_e = [], []
                    for _, g in eps_.groupby("y"):
                        idx = rng.permutation(len(g)); a = int(round(0.7 * len(g)))
                        tr_e += list(g.episode.iloc[idx[:a]]); ca_e += list(g.episode.iloc[idx[a:]])
                    tr, ca = pool[pool.episode.isin(tr_e)], pool[pool.episode.isin(ca_e)]; h0 = tr[tr.y == 0]
                    thr = (float(np.percentile(h0.ue_lat, 90)), float(np.percentile(h0.amf_lat, 90)))
                    E_tr, E_ca, E_te = episodes(tr, thr), episodes(ca, thr), episodes(te, thr)
                    yca = np.array([y for _, y, _ in E_ca]); yte = np.array([y for _, y, _ in E_te])
                    r1 = np.random.default_rng(SEED * 3 + seed + held + 10007 * k[0] + 31 * k[1])
                    W1 = [make_window(z, y, 1, 0.0, r1) for _, y, z in E_tr]; Wc1 = [make_window(z, y, 1, 0.0, r1) for _, y, z in E_ca]
                    st = fit([L.count_features(w) for w in W1], [w.y for w in W1])
                    gate.append(dict(cond=cond, task=task, held_block=held, seed=seed,
                                     auc_cal=M.auroc(st.predict_proba(np.array([L.count_features(w) for w in Wc1]))[:, 1], yca)))
                    Wa, Wc = [], []
                    for rep in range(5):
                        Wa += [make_window(z, y, int(r1.choice(AUG_D)), float(r1.choice(H_GRID)), r1) for _, y, z in E_tr]
                    for rep in range(3):
                        Wc += [make_window(z, y, int(r1.choice(AUG_D)), float(r1.choice(H_GRID)), r1) for _, y, z in E_ca]
                    ya, yc = [w.y for w in Wa], np.array([w.y for w in Wc])
                    Ca, Cc = np.array([L.count_features(w) for w in Wa]), np.array([L.count_features(w) for w in Wc])
                    Pa, Pc = np.array([L.presence_features(w) for w in Wa]), np.array([L.presence_features(w) for w in Wc])
                    Fa, Fc = np.array([lineage_separated_features(L, w) for w in Wa]), np.array([lineage_separated_features(L, w) for w in Wc])
                    Ua, Uc = np.array([unit_feats(w) for w in Wa]), np.array([unit_feats(w) for w in Wc])
                    aug = fit(Ca, ya)
                    gb_aug = gb_fit(Ca, ya, Cc, yc); gb_dd = gb_fit(Pa, ya, Pc, yc)
                    gb_la = gb_fit(Fa, ya, Fc, yc); gb_unit = gb_fit(Ua, ya, Uc, yc)
                    r3 = np.random.default_rng(SEED * 13 + seed * 101 + held * 1000 + k[1] * 7 + 10007 * k[0])
                    Wmix = [make_window_r10(z, y, m, 0.5, r3) for m in LD_PAIR for _, y, z in E_tr for _ in range(3)]
                    Wmixc = [make_window_r10(z, y, m, 0.5, r3) for m in LD_PAIR for _, y, z in E_ca for _ in range(2)]
                    gb_mix = gb_fit(np.array([L.count_features(w) for w in Wmix]), [w.y for w in Wmix],
                                    np.array([L.count_features(w) for w in Wmixc]), np.array([w.y for w in Wmixc]))
                    for mech in MECHS:
                        W = [make_window_r10(z, y, mech, 0.5, r3) for _, y, z in E_te]
                        C = np.array([L.count_features(w) for w in W]); P = np.array([L.presence_features(w) for w in W])
                        F = np.array([lineage_separated_features(L, w) for w in W]); U = np.array([unit_feats(w) for w in W])
                        rows.append(dict(cond=cond, task=task, held_block=held, seed=seed, mech=mech,
                                         arm="none" if mech == "none" else ("LI" if mech in LI else "LD"), n=len(yte),
                                         augmented=M.brier(aug.predict_proba(C)[:, 1], yte),
                                         gbdt_aug=M.brier(gb_aug.predict_proba(C)[:, 1], yte),
                                         gbdt_dedup=M.brier(gb_dd.predict_proba(P)[:, 1], yte),
                                         gbdt_la=M.brier(gb_la.predict_proba(F)[:, 1], yte),
                                         gbdt_unit=M.brier(gb_unit.predict_proba(U)[:, 1], yte),
                                         gbdt_mix_oracle=M.brier(gb_mix.predict_proba(C)[:, 1], yte),
                                         logloss_unit=M.nll(gb_unit.predict_proba(U)[:, 1], yte),
                                         logloss_mix=M.nll(gb_mix.predict_proba(C)[:, 1], yte),
                                         auroc_unit=M.auroc(gb_unit.predict_proba(U)[:, 1], yte),
                                         auroc_mix=M.auroc(gb_mix.predict_proba(C)[:, 1], yte)))
                print(cond, task, held, flush=True)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "r11_cells.csv"), index=False)
    pd.DataFrame(gate).to_csv(os.path.join(out, "r11_gate.csv"), index=False)


if __name__ == "__main__":
    main()
