"""A2c: leakage-free parameters + credal box for FRESH origin parameters (exploratory)."""
from __future__ import annotations
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
from run_pilot import score_episode, summarise  # noqa: E402
from paef.baselines.simple import oracle  # noqa: E402
from paef.simulation.agnostic import make_agnostic  # noqa: E402
from paef.simulation.generator import GenConfig, sample_episode  # noqa: E402
SEED = 20261017

def job(args):
    seed, kw = args
    ep = sample_episode(np.random.default_rng(seed), GenConfig(**kw))
    row = score_episode(make_agnostic(ep, fresh_box=True))
    row["oracle"] = oracle(ep)
    return row

def main():
    rep = {}
    cells = [(dict(hide=0.5), lambda s: SEED * 1000 + 97 * 2 + s)] + \
            [(dict(hide=0.75, p_true_fresh=pf), (lambda j: (lambda s: SEED * 7 + 131 * j + s))(j))
             for j, pf in enumerate((0.0, 0.33, 0.67, 1.0))]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for kw, sf in cells:
            rows = list(ex.map(job, [(sf(s), kw) for s in range(3000)], chunksize=20))
            rep[json.dumps(kw)] = summarise(rows)
            print(kw, flush=True)
    json.dump(rep, open(os.path.join(HERE, "..", "..", "results/pilot/amendment_a2c.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
