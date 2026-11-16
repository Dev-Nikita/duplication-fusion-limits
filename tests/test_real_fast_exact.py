"""The fast typed computation in paef.real.origami equals generic enumeration (paef.fusion.bounds)."""
import numpy as np
import pytest

from paef.fusion.bounds import posterior_bounds
from paef.model.lineage import COPY, FRESH, Episode, Observation, log_odds_given_assignment, sigmoid
from paef.real.origami import NODES, OriginModel, Window, fuse


def to_episode(w, models):
    ep = Episode(prior=0.5, theta={i: tuple(models[n].theta) for i, n in enumerate(NODES) if n in w.z}, label=w.y)
    idx = {}
    true_assign = {}
    for i, n in enumerate(NODES):
        if n not in w.z:
            continue
        m = models[n]
        for k in range(3):
            hid = k in w.hidden_read.get(n, [])
            cand = (i, FRESH) if hid else (i,)
            ep.obs.append(Observation(int(w.z[n][k]), (m.c[k, 0], m.c[k, 1]), cand, tuple(m.theta), i, i))
            idx[(n, k)] = len(ep.obs) - 1
    for (n, k, hid) in w.copies:
        p = idx[(n, k)]
        o = ep.obs[p]
        cand = (COPY(p), FRESH) if hid else (COPY(p),)
        ep.obs.append(Observation(1, o.chan, cand, o.fresh_theta, o.true_root, COPY(p), "exact"))
    return ep


@pytest.mark.parametrize("seed", range(40))
def test_fast_equals_generic(seed):
    rng = np.random.default_rng(seed)
    models = {n: OriginModel(theta=np.sort(rng.uniform(0.05, 0.95, 2)),
                             c=np.sort(rng.uniform(0.02, 0.98, (3, 2)), axis=1)) for n in NODES}
    nodes = list(rng.choice(NODES, size=2, replace=False))
    z = {n: rng.integers(0, 2, 3) for n in nodes}
    w = Window(y=1, z=z, hidden_read={n: [k for k in (1, 2) if rng.random() < 0.6] for n in nodes})
    for n in nodes:
        for k in range(3):
            if z[n][k]:
                for _ in range(int(rng.integers(0, 3))):
                    w.copies.append((n, k, bool(rng.random() < 0.6)))
    r = fuse(w, models)
    ep = to_episode(w, models)
    b = posterior_bounds(ep)
    assert abs(r["paef_lower"] - b["lower"]) < 1e-9 and abs(r["paef_upper"] - b["upper"]) < 1e-9
    assert abs(r["paef_bma"] - b["bma"]) < 1e-9
    orc = sigmoid(log_odds_given_assignment(ep, {i: o.true_code for i, o in enumerate(ep.obs)}))
    assert abs(r["oracle"] - orc) < 1e-9
