"""Amendment A2: leakage audit (see docs/amendments.md).

A2a static: fusion and baseline code must not read ground-truth fields except `oracle`.
A2b dynamic: re-run P2 (hide 0.5, 0.75) and A1b (p_true_fresh 0, 1) with the leakage-free
     parameterisation of paef.simulation.agnostic; compare with the original numbers.
The oracle keeps the TRUE parameters (it is the reference), so it is scored on the original episode.
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from run_pilot import score_episode, summarise  # noqa: E402
from paef.baselines.simple import oracle  # noqa: E402
from paef.simulation.agnostic import make_agnostic  # noqa: E402
from paef.simulation.generator import GenConfig, sample_episode  # noqa: E402

SEED = 20261017
ROOT = os.path.join(HERE, "..", "..")


def static_audit():
    hits = []
    for rel in ("src/paef/fusion/bounds.py", "src/paef/baselines/simple.py"):
        for n, line in enumerate(open(os.path.join(ROOT, rel)), 1):
            if re.search(r"\.(true_root|true_code|label)\b", line):
                hits.append(f"{rel}:{n}: {line.strip()}")
    return hits


def job(args):
    seed, kw = args
    ep = sample_episode(np.random.default_rng(seed), GenConfig(**kw))
    row = score_episode(make_agnostic(ep))
    row["oracle"] = oracle(ep)
    return row


def main():
    rep = {"A2a_static_hits": static_audit()}
    cells = [("P2", dict(hide=0.5), lambda s: SEED * 1000 + 97 * 2 + s),
             ("P2", dict(hide=0.75), lambda s: SEED * 1000 + 97 * 3 + s),
             ("A1b", dict(hide=0.75, p_true_fresh=0.0), lambda s: SEED * 7 + 131 * 0 + s),
             ("A1b", dict(hide=0.75, p_true_fresh=1.0), lambda s: SEED * 7 + 131 * 3 + s)]
    rep["A2b"] = {}
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for tag, kw, sf in cells:
            rows = list(ex.map(job, [(sf(s), kw) for s in range(3000)], chunksize=20))
            rep["A2b"][tag + " " + json.dumps(kw)] = summarise(rows)
            print(tag, kw, flush=True)
    with open(os.path.join(ROOT, "results/pilot/amendment_a2.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
