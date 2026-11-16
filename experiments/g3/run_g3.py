"""G3 pilot (protocol docs/g3-protocol.md; theory docs/theory/G3_FORMALIZATION.md)."""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "g2"))

import run_g2 as g2  # noqa: E402
from paef.baselines.simple import oracle  # noqa: E402
from paef.calibration.conformal_sets import build_sets, conformal_quantile, episode_scores  # noqa: E402
from paef.calibration.scorer import LineageScorer, node_probs  # noqa: E402
from paef.g3.hypotheses import build_table, status  # noqa: E402
from paef.g3.policies import POLICIES, dp_optimal_value, run_policy  # noqa: E402
from paef.simulation.agnostic import make_agnostic  # noqa: E402

SEED = 20261019


def rep(args):
    r, hide = args
    rng = np.random.default_rng(SEED * 100 + r * 13 + int(hide * 100))
    cfg = g2.cfg_for(hide)
    train, cal, test = g2.draw(rng, cfg, 2000), g2.draw(rng, cfg, 1000), g2.draw(rng, cfg, 400)
    sc = LineageScorer().fit(train)
    q = conformal_quantile(episode_scores(sc, cal, typed=False), 0.05)
    rows = []
    for ep in test:
        S = build_sets(sc, ep, q, False)
        if not S:
            continue
        P = node_probs(sc, ep, typed=False)
        probs = {j: dict(zip(ep.meta[j][0], P[j])) for j in P}
        e = make_agnostic(ep, fresh_box=True)
        for j, s in S.items():
            e.obs[j].candidates = s
        t = build_table(e, S, probs)
        if len(t.lo) == 0:
            continue
        full = np.ones(len(t.lo), dtype=bool)
        if status(t, full) != -1:
            continue  # decided without queries: not in the population
        truth = np.array([ep.obs[j].true_code for j in t.nodes])
        p_star = oracle(ep)
        row = {"n_unres": len(t.nodes), "n_hyp": len(t.lo), "single": t.single_per_cluster,
               "y": ep.label, "orc_dec": int(p_star > 0.5)}
        prng = np.random.default_rng(r * 7919 + len(rows))
        for name, pol in POLICIES.items():
            out = run_policy(t, truth, pol, prng)
            row[name] = out
        row["dp_optimal"] = dp_optimal_value(t)
        rows.append(row)
    return rows


def boot_diff(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summarise(rows):
    out = {"n_undecided_episodes": len(rows),
           "share_single_per_cluster": float(np.mean([r["single"] for r in rows])),
           "mean_unresolved": float(np.mean([r["n_unres"] for r in rows]))}
    for name in POLICIES:
        q = np.array([r[name]["queries"] for r in rows])
        dec = np.array([r[name]["decision"] for r in rows])
        miss = np.array([r[name]["set_miss"] for r in rows])
        y = np.array([r["y"] for r in rows]); od = np.array([r["orc_dec"] for r in rows])
        d = dec >= 0
        out[name] = {"mean_queries": float(q.mean()), "decided": float(d.mean()),
                     "set_miss": float(miss.mean()),
                     "err_vs_label": float(np.mean(dec[d] != y[d])) if d.any() else float("nan"),
                     "disagree_vs_oracle_decision": float(np.mean(dec[d] != od[d])) if d.any() else float("nan")}
    small = [r for r in rows if r["dp_optimal"] is not None]
    out["dp_subset_n"] = len(small)
    out["dp_optimal_mean_queries_expected"] = float(np.mean([r["dp_optimal"] for r in small])) if small else None
    for name in POLICIES:
        out[name]["mean_queries_on_dp_subset"] = float(np.mean([r[name]["queries"] for r in small])) if small else None
    for a in ("decision_robustness", "ec2"):
        for b in ("random", "max_uncertainty"):
            out[f"diff_{a}_minus_{b}"] = boot_diff([r[a]["queries"] for r in rows], [r[b]["queries"] for r in rows])
    out["diff_ec2_minus_decision_robustness"] = boot_diff([r["ec2"]["queries"] for r in rows],
                                                          [r["decision_robustness"]["queries"] for r in rows])
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for hide in (0.5, 0.75):
            rows = [x for rr in ex.map(rep, [(r, hide) for r in range(reps)]) for x in rr]
            res[f"hide={hide}"] = summarise(rows)
            print(hide, len(rows), round(time.time() - t0), flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(ROOT, "results/g3"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results/g3/pilot.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
