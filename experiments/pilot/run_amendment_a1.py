"""Amendment A1 (exploratory, added after the P1-P4 run; see docs/amendments.md).

A1a  P1 re-analysed by the sign of the copied evidence: which bound is invariant?
A1b  P5: hidden secondaries are truly independent with probability p_true_fresh.
     Tests whether paef_bma's advantage depends on its uniform lineage prior matching
     the generator, and whether the interval keeps its guarantee when it does not.
A1c  reproducibility check: re-run P2 hide=0.5 with the amended generator and compare.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from run_pilot import POINT, _sweep_job, score_episode, summarise  # noqa: E402
from paef.model.lineage import COPY, FRESH, Observation  # noqa: E402
from paef.simulation.generator import GenConfig, sample_episode  # noqa: E402

SEED = 20261017


def a1a(n=400):
    rng = np.random.default_rng(SEED + 1)  # same stream as P1
    out = {}
    for _ in range(n):
        ep0 = sample_episode(rng, GenConfig(hide=0.0, sec_per_root=(0, 1)))
        base = score_episode(ep0)
        sign = "copy_of_z1" if ep0.obs[0].value == 1 else "copy_of_z0"
        for m, cand in (("unknown_fresh", (COPY(0), FRESH)), ("unknown_decoy", (COPY(0), COPY(1), FRESH))):
            for k in (1, 2, 4, 8):
                ep = copy.deepcopy(ep0)
                for _ in range(k):
                    o0 = ep.obs[0]
                    ep.obs.append(Observation(o0.value, o0.chan, cand, o0.fresh_theta, 0, COPY(0), "exact"))
                row = score_episode(ep)
                for key in ("paef_lower", "paef_upper", "paef_bma", "naive_bayes"):
                    out.setdefault(f"{sign}|{m}|{k}|{key}", []).append(row[key] - base[key])
    return {k: {"mean_signed": float(np.mean(v)), "max_abs": float(np.max(np.abs(v))), "n": len(v)}
            for k, v in out.items()}


def main():
    out_dir = "results/pilot"
    rep = {"A1a_sign_split": a1a()}
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        rep["A1b_true_fresh"] = {}
        for j, pf in enumerate((0.0, 0.33, 0.67, 1.0)):
            kw = dict(hide=0.75, p_true_fresh=pf)
            rows = list(ex.map(_sweep_job, [(SEED * 7 + 131 * j + s, kw) for s in range(3000)], chunksize=20))
            rep["A1b_true_fresh"][json.dumps(kw)] = summarise(rows)
            print("A1b", kw, flush=True)
        kw = dict(hide=0.5)
        rows = list(ex.map(_sweep_job, [(SEED * 1000 + 97 * 2 + s, kw) for s in range(3000)], chunksize=20))
        rep["A1c_repro_hide0.5"] = summarise(rows)
    with open(os.path.join(out_dir, "amendment_a1.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
