"""Amendment A3 (exploratory, added after seeing G2-P): posterior-targeted nested calibration.

Observation from G2-P: posterior-envelope coverage (~0.99) far exceeds lineage coverage (~0.95), i.e.
calibrating the lineage is conservative for the object we actually report. Nested conformal
(sets indexed by q are nested; envelope [L(q),U(q)] grows with q):
  q*(e) = min{ q in grid : p*(e) ∈ [L_e(q), U_e(q)] }   on calibration episodes,
  q̂ = conformal quantile of q*;  test envelope = [L(q̂), U(q̂)].
Exchangeability + nestedness give P(p* ∈ [L(q̂),U(q̂)]) ≥ 1−δ for the true-model posterior p*
(object 2) — NOT for the lineage (object 1). Requires p* on calibration episodes, i.e. known lineage and
parameters there (the pilot has them; real data must supply lineage by construction).
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
import run_g2 as g  # noqa: E402
from paef.baselines.simple import oracle  # noqa: E402
from paef.calibration.conformal_sets import build_sets, conformal_quantile  # noqa: E402
from paef.calibration.scorer import LineageScorer  # noqa: E402

GRID = np.r_[np.linspace(0.0, 0.95, 39), 0.99, 1.0]


def qstar(sc, ep):
    p = oracle(ep)
    for q in GRID:
        L, U, _ = g.envelope(ep, build_sets(sc, ep, q, False))
        if L - 1e-9 <= p <= U + 1e-9:
            return float(q)
    return 1.0


def rep(args):
    r, hide = args
    rng = np.random.default_rng(g.SEED * 100 + r * 11 + int(hide * 100) + 10)  # same stream as G2-P
    cfg = g.cfg_for(hide)
    train, cal, test = g.draw(rng, cfg, 2000), g.draw(rng, cfg, 1000), g.draw(rng, cfg, 300)
    sc = LineageScorer().fit(train)
    q_hat = conformal_quantile([qstar(sc, ep) for ep in cal], 0.05)
    cov, w, dec = [], [], []
    for ep in test:
        L, U, _ = g.envelope(ep, build_sets(sc, ep, q_hat, False))
        p = oracle(ep)
        cov.append(L - 1e-9 <= p <= U + 1e-9); w.append(U - L); dec.append((L > .5) or (U < .5))
    return {"q_hat": q_hat, "post_cov": float(np.mean(cov)), "width": float(np.mean(w)), "decided": float(np.mean(dec))}


def main():
    out = {}
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for hide in (0.5, 0.75):
            rs = list(ex.map(rep, [(r, hide) for r in range(20)]))
            out[f"hide={hide}"] = {k: {"mean": float(np.mean([x[k] for x in rs])),
                                       "ci95": [float(np.percentile([x[k] for x in rs], 2.5)),
                                                float(np.percentile([x[k] for x in rs], 97.5))]} for k in rs[0]}
            print(hide, out[f"hide={hide}"], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results/g2/amendment_a3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
