"""Prop. 1 (one-sided invariance) on random episodes including conflicting readings and mixed values.
Adds an observation that may be a copy of observed parent p (same value) or any root reading or FRESH."""
import copy

import numpy as np
import pytest

from paef.fusion.bounds import posterior_bounds
from paef.model.lineage import COPY, FRESH, Observation
from paef.simulation.generator import GenConfig, sample_episode


@pytest.mark.parametrize("seed", range(300))
def test_opposite_bound_invariant(seed):
    rng = np.random.default_rng(10_000 + seed)
    ep = sample_episode(rng, GenConfig(hide=0.6, p_true_fresh=0.3, k_roots=(2, 3), sec_per_root=(1, 2)))
    obs = [i for i in ep.observed()]
    p = int(rng.choice(obs))
    b0 = posterior_bounds(ep)
    e = copy.deepcopy(ep)
    o = e.obs[p]
    cands = [COPY(p)] + [int(r) for r in e.theta] + [FRESH]
    e.obs.append(Observation(o.value, o.chan, tuple(cands), o.fresh_theta, o.true_root, COPY(p), "exact"))
    b1 = posterior_bounds(e)
    if o.value == 1:
        assert abs(b1["lo_logodds"] - b0["lo_logodds"]) < 1e-9
        assert b1["hi_logodds"] >= b0["hi_logodds"] - 1e-9
    else:
        assert abs(b1["hi_logodds"] - b0["hi_logodds"]) < 1e-9
        assert b1["lo_logodds"] <= b0["lo_logodds"] + 1e-9
