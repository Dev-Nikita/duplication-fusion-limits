"""Pilot P1-P4 (synthetic). Protocol: docs/pilot-protocol.md (written before this run).

Usage:  python -I experiments/pilot/run_pilot.py --out results/pilot --n 1500
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from paef.baselines.simple import BASELINES, oracle  # noqa: E402
from paef.evaluation import metrics as M  # noqa: E402
from paef.fusion.bounds import posterior_bounds, resolve_value  # noqa: E402
from paef.model.lineage import COPY, FRESH, Observation  # noqa: E402
from paef.simulation.generator import GenConfig, sample_episode  # noqa: E402

POINT = ["naive_bayes", "declared_only", "pact_meet_declared", "pact_meet_closure",
         "cautious_closure", "paef_bma", "oracle"]


def score_episode(ep):
    row = {k: f(ep) for k, f in BASELINES.items()}
    b = posterior_bounds(ep)
    row.update(paef_bma=b["bma"], paef_lower=b["lower"], paef_upper=b["upper"],
               n_assign=b["n_assign"], y=ep.label)
    return row


def _sweep_job(args):
    seed, cfg_kw = args
    rng = np.random.default_rng(seed)
    ep = sample_episode(rng, GenConfig(**cfg_kw))
    return score_episode(ep)


def summarise(rows, t=0.5):
    y = np.array([r["y"] for r in rows])
    out = {"n": len(rows), "prevalence": float(y.mean())}
    lo = np.array([r["paef_lower"] for r in rows])
    hi = np.array([r["paef_upper"] for r in rows])
    orc = np.array([r["oracle"] for r in rows])
    decided = (lo > t) | (hi < t)
    pred_int = (lo > t).astype(int)
    cov = float(decided.mean())
    out["paef_interval"] = {
        "coverage_decided": cov,
        "risk_decided": float(np.mean(pred_int[decided] != y[decided])) if decided.any() else float("nan"),
        "mean_width": float(np.mean(hi - lo)),
        "oracle_in_interval": float(np.mean((orc >= lo - 1e-9) & (orc <= hi + 1e-9))),
        "aurc_margin": M.aurc(np.maximum(lo - t, t - hi), (lo + hi > 2 * t).astype(int), y),
    }
    for k in POINT:
        p = np.array([r[k] for r in rows])
        pred = (p > t).astype(int)
        conf = np.abs(p - t)
        out[k] = {"brier": M.brier(p, y), "nll": M.nll(p, y), "ece": M.ece(p, y),
                  "auroc": M.auroc(p, y), "err": float(np.mean(pred != y)),
                  "risk_at_paef_cov": M.risk_at_coverage(conf, pred, y, cov) if cov > 0 else float("nan"),
                  "aurc": M.aurc(conf, pred, y)}
    return out


# ---------------------------------------------------------------- P1 duplication
def p1_duplication(n, seed):
    rng = np.random.default_rng(seed)
    ks = [1, 2, 4, 8]
    modes = ["known", "unknown_fresh", "unknown_decoy"]
    res = {m: {k: {} for k in ks} for m in modes}
    acc = {m: {k: {} for k in ks} for m in modes}
    for _ in range(n):
        ep0 = sample_episode(rng, GenConfig(hide=0.0, sec_per_root=(0, 1)))
        base = score_episode(ep0)
        p = 0  # primary of root 0
        decoy_p = 1
        for m in modes:
            for k in ks:
                ep = copy.deepcopy(ep0)
                for _ in range(k):
                    if m == "known":
                        c = (COPY(p),)
                    elif m == "unknown_fresh":
                        c = (COPY(p), FRESH)
                    else:
                        c = (COPY(p), COPY(decoy_p), FRESH)
                    o0 = ep.obs[p]
                    ep.obs.append(Observation(o0.value, o0.chan, c, o0.fresh_theta, 0, COPY(p), "exact"))
                row = score_episode(ep)
                for key in POINT + ["paef_lower", "paef_upper"]:
                    acc[m][k].setdefault(key, []).append(abs(row[key] - base[key]))
    for m in modes:
        for k in ks:
            for key, v in acc[m][k].items():
                res[m][k][key] = {"mean": float(np.mean(v)), "max": float(np.max(v))}
    return res


# ---------------------------------------------------------------- P4 value of provenance info
def p4_resolution(n, seed, hide=0.75):
    rng = np.random.default_rng(seed)
    out = {"vopi": [], "random": []}
    tried = 0
    while len(out["vopi"]) < n and tried < 50 * n:
        tried += 1
        ep = sample_episode(rng, GenConfig(hide=hide))
        b = posterior_bounds(ep)
        if not (b["lower"] < 0.5 < b["upper"]):
            continue
        unres = [i for i in ep.observed() if len(ep.obs[i].candidates) > 1]
        if len(unres) < 2:
            continue
        vals = [resolve_value(ep, i) for i in unres]
        for policy in ("vopi", "random"):
            i = unres[int(np.argmax(vals))] if policy == "vopi" else unres[int(rng.integers(len(unres)))]
            e2 = copy.deepcopy(ep)
            e2.obs[i].candidates = (e2.obs[i].true_code,)  # an operator resolves this link
            b2 = posterior_bounds(e2)
            decided = (b2["lower"] > 0.5) or (b2["upper"] < 0.5)
            out[policy].append({"decided": decided,
                                "correct": decided and int(b2["lower"] > 0.5) == ep.label,
                                "width_logodds_before": b["hi_logodds"] - b["lo_logodds"],
                                "width_logodds_after": b2["hi_logodds"] - b2["lo_logodds"]})
    summ = {}
    for policy, rows in out.items():
        d = np.array([r["decided"] for r in rows])
        c = np.array([r["correct"] for r in rows])
        red = np.array([r["width_logodds_before"] - r["width_logodds_after"] for r in rows])
        summ[policy] = {"n": len(rows), "decided_after_one": float(d.mean()),
                        "decided_ci": M.bootstrap_ci(d.astype(float)),
                        "risk_among_decided": float(1 - c[d].mean()) if d.any() else float("nan"),
                        "mean_width_reduction": float(red.mean()),
                        "width_reduction_ci": M.bootstrap_ci(red)}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pilot")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    report = {"seed": a.seed, "n": a.n}

    report["P1_duplication"] = p1_duplication(min(a.n, 400), a.seed + 1)
    print("P1 done", round(time.time() - t0, 1), flush=True)

    sweeps = {"P2_hide": [dict(hide=h) for h in (0.0, 0.25, 0.5, 0.75, 1.0)],
              "P3_misspec": [dict(hide=0.5, misspec=m) for m in (0.0, 0.1, 0.3, 0.5)]}
    with ProcessPoolExecutor(a.workers) as ex:
        for name, cfgs in sweeps.items():
            report[name] = {}
            for j, kw in enumerate(cfgs):
                seeds = [(a.seed * 1000 + 97 * j + s, kw) for s in range(a.n)]
                rows = list(ex.map(_sweep_job, seeds, chunksize=20))
                report[name][json.dumps(kw)] = summarise(rows)
                print(name, kw, "done", round(time.time() - t0, 1), flush=True)

    report["P4_resolution"] = p4_resolution(min(a.n, 300), a.seed + 4)
    print("P4 done", round(time.time() - t0, 1), flush=True)
    report["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "pilot_report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
