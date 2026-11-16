"""G2 pilot (protocol: docs/g2-protocol.md; theory: docs/theory/G2_coverage.md).

Usage: python -I experiments/g2/run_g2.py --exp L|P|S [--reps N]
"""
from __future__ import annotations

import argparse
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

from paef.baselines.simple import oracle  # noqa: E402
from paef.calibration.conformal_sets import (build_sets, conformal_quantile, covers,  # noqa: E402
                                             episode_scores, node_scores)
from paef.calibration.scorer import LineageScorer, node_probs  # noqa: E402
from paef.fusion.bounds import posterior_bounds, weighted_posterior  # noqa: E402
from paef.model.lineage import log_odds_given_assignment, sigmoid  # noqa: E402
from paef.simulation.agnostic import make_agnostic  # noqa: E402
from paef.simulation.generator import GenConfig  # noqa: E402
from paef.simulation.lineage_features import FeatConfig, sample_featured  # noqa: E402

SEED = 20261018


def cfg_for(hide, lag_true=1.0):
    return FeatConfig(gen=GenConfig(hide=hide, p_true_fresh=0.3, k_roots=(2, 3), sec_per_root=(1, 2)),
                      lag_true=lag_true)


def draw(rng, cfg, n):
    return [sample_featured(rng, cfg) for _ in range(n)]


def all_sets(scorer, cal, delta):
    """Thresholds for every method from the calibration split."""
    q_ep = conformal_quantile(episode_scores(scorer, cal, typed=False), delta)
    q_ep_t = conformal_quantile(episode_scores(scorer, cal, typed=True), delta)
    ns, mJ = node_scores(scorer, cal, typed=False)
    q_marg = conformal_quantile(ns, delta)
    q_bonf = conformal_quantile(ns, delta / mJ)
    return {"marginal_sets": (q_marg, False), "bonferroni_sets": (q_bonf, False),
            "paef_cal": (q_ep, False), "paef_cal_typed": (q_ep_t, True)}


# ------------------------------------------------------------------ Experiment L
def rep_L(args):
    rep, hide = args
    rng = np.random.default_rng(SEED * 10 + rep * 7 + int(hide * 100))
    cfg = cfg_for(hide)
    train, cal_all, test = draw(rng, cfg, 2000), draw(rng, cfg, 2000), draw(rng, cfg, 1000)
    sc = LineageScorer().fit(train)
    out = {}
    for n_cal in (250, 500, 1000, 2000):
        cal = cal_all[:n_cal]
        for delta in (0.10, 0.05):
            th = all_sets(sc, cal, delta)
            for m, (q, typed) in th.items():
                cov, size = [], []
                for ep in test:
                    S = build_sets(sc, ep, q, typed)
                    cov.append(covers(ep, S))
                    size.append(np.mean([len(s) for s in S.values()]) if S else np.nan)
                out[f"{n_cal}|{delta}|{m}"] = (float(np.mean(cov)), float(np.nanmean(size)))
    return out


# ------------------------------------------------------------------ Experiment P / S
def envelope(ep, sets):
    e = copy.deepcopy(ep)
    for j, s in sets.items():
        e.obs[j].candidates = s
    e = make_agnostic(e, fresh_box=True)
    t = time.perf_counter()
    b = posterior_bounds(e)
    return b["lower"], b["upper"], time.perf_counter() - t


def rep_P(args):
    rep, hide, lag_test = args
    rng = np.random.default_rng(SEED * 100 + rep * 11 + int(hide * 100) + int(lag_test * 10))
    cfg = cfg_for(hide)
    train, cal = draw(rng, cfg, 2000), draw(rng, cfg, 1000)
    test = draw(rng, cfg_for(hide, lag_true=lag_test), 300)
    sc = LineageScorer().fit(train)
    th = all_sets(sc, cal, 0.05)
    rows = []
    for ep in test:
        p_star = oracle(ep)
        r = {"y": ep.label, "oracle": p_star, "n_unres": len([j for j in ep.meta if ep.obs[j].value is not None])}
        full = {j: ep.meta[j][0] for j in ep.meta if ep.obs[j].value is not None}
        L, U, t = envelope(ep, full)
        r["cma_full"] = (L, U, t, True)
        for m, (q, typed) in th.items():
            S = build_sets(sc, ep, q, typed)
            L, U, t = envelope(ep, S)
            r[m] = (L, U, t, covers(ep, S))
        # predictive outputs (scorer-weighted average; argmax plug-in), leakage-free parameters
        P = node_probs(sc, ep, typed=True)
        e = make_agnostic(ep)
        W = {j: dict(zip(ep.meta[j][0], p)) for j, p in P.items()}
        r["bma_scored"] = weighted_posterior(e, W)
        assign = {}
        for i in e.observed():
            assign[i] = ep.meta[i][0][int(np.argmax(P[i]))] if i in P else e.obs[i].candidates[0]
        try:
            r["argmax_plugin"] = float(sigmoid(log_odds_given_assignment(e, assign)))
        except TypeError:
            r["argmax_plugin"] = float("nan")
        rows.append(r)
    return rows


def summarise_P(all_rows):
    methods = ["cma_full", "marginal_sets", "bonferroni_sets", "paef_cal", "paef_cal_typed"]
    reps = []
    for rows in all_rows:
        d = {}
        for m in methods:
            L = np.array([r[m][0] for r in rows]); U = np.array([r[m][1] for r in rows])
            ps = np.array([r["oracle"] for r in rows]); y = np.array([r["y"] for r in rows])
            decided = (L > 0.5) | (U < 0.5)
            d[m] = {"post_cov": float(np.mean((ps >= L - 1e-9) & (ps <= U + 1e-9))),
                    "lineage_cov": float(np.mean([r[m][3] for r in rows])),
                    "width": float(np.mean(U - L)),
                    "decided": float(decided.mean()),
                    "risk_decided": float(np.mean((L[decided] > 0.5).astype(int) != y[decided])) if decided.any() else float("nan"),
                    "time_ms": float(1000 * np.mean([r[m][2] for r in rows]))}
        y = np.array([r["y"] for r in rows])
        for k in ("bma_scored", "argmax_plugin", "oracle"):
            p = np.array([r[k] for r in rows])
            d[k] = {"brier": float(np.nanmean((p - y) ** 2))}
        reps.append(d)
    out = {}
    for m in reps[0]:
        out[m] = {}
        for k in reps[0][m]:
            v = np.array([r[m][k] for r in reps], dtype=float)
            out[m][k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                         "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
    # paired width differences
    def paired(a, b):
        dv = np.array([r[a]["width"] - r[b]["width"] for r in reps])
        return {"mean": float(dv.mean()), "se": float(dv.std(ddof=1) / np.sqrt(len(dv))),
                "frac_negative": float(np.mean(dv < 0))}
    out["paired_width_typed_minus_cal"] = paired("paef_cal_typed", "paef_cal")
    out["width_reduction_typed_vs_cma"] = float(1 - out["paef_cal_typed"]["width"]["mean"] / out["cma_full"]["width"]["mean"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["L", "P", "S"])
    ap.add_argument("--reps", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results/g2"), exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        if a.exp == "L":
            reps = a.reps or 200
            res = {}
            for hide in (0.5, 0.75):
                outs = list(ex.map(rep_L, [(r, hide) for r in range(reps)]))
                agg = {}
                for k in outs[0]:
                    cov = np.array([o[k][0] for o in outs]); size = np.array([o[k][1] for o in outs])
                    agg[k] = {"cov_mean": float(cov.mean()), "cov_se": float(cov.std(ddof=1) / np.sqrt(len(cov))),
                              "cov_p05": float(np.percentile(cov, 5)), "size_mean": float(size.mean())}
                res[f"hide={hide}"] = agg
                print("L hide", hide, round(time.time() - t0), flush=True)
        else:
            reps = a.reps or 20
            res = {}
            settings = [(0.5, 1.0), (0.75, 1.0)] if a.exp == "P" else [(0.5, 1.6)]
            for hide, lag in settings:
                outs = list(ex.map(rep_P, [(r, hide, lag) for r in range(reps)]))
                res[f"hide={hide},lag_test={lag}"] = summarise_P(outs)
                print(a.exp, hide, lag, round(time.time() - t0), flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(ROOT, f"results/g2/exp_{a.exp}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
