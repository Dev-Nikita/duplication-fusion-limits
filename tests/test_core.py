import numpy as np
import pytest

from paef.model.lineage import COPY, FRESH, Episode, Observation
from paef.fusion.bounds import posterior_bounds
from paef.baselines.simple import naive_bayes, oracle, pact_meet_declared
from paef.simulation.generator import GenConfig, sample_episode


def _base():
    ep = Episode(prior=0.3, theta={0: (0.1, 0.9), 1: (0.2, 0.8)}, label=1)
    ep.obs.append(Observation(1, (0.1, 0.9), (0,), (0.1, 0.9), 0, 0))
    ep.obs.append(Observation(1, (0.1, 0.9), (1,), (0.2, 0.8), 1, 1))
    return ep


def test_known_exact_copy_is_invariant():
    ep = _base()
    p0 = oracle(ep)
    for _ in range(8):
        ep.obs.append(Observation(1, (0.1, 0.9), (COPY(0),), (0.1, 0.9), 0, COPY(0), "exact"))
    b = posterior_bounds(ep)
    assert abs(b["lower"] - p0) < 1e-12 and abs(b["upper"] - p0) < 1e-12
    assert abs(oracle(ep) - p0) < 1e-12
    assert naive_bayes(ep) > p0 + 0.05  # naive fusion amplifies


def test_unknown_copy_lower_bound_invariant_upper_grows():
    ep = _base()
    p0 = oracle(ep)
    uppers = []
    for _ in range(6):
        ep.obs.append(Observation(1, (0.1, 0.9), (COPY(0), FRESH), (0.1, 0.9), 0, COPY(0), "exact"))
        b = posterior_bounds(ep)
        assert abs(b["lower"] - p0) < 1e-9
        uppers.append(b["upper"])
    assert all(np.diff(uppers) >= -1e-12)


@pytest.mark.parametrize("seed", range(30))
def test_oracle_inside_interval_when_well_specified(seed):
    rng = np.random.default_rng(seed)
    cfg = GenConfig(k_roots=(2, 3), sec_per_root=(1, 3), hide=0.7, misspec=0.0)
    ep = sample_episode(rng, cfg)
    b = posterior_bounds(ep)
    p = oracle(ep)
    assert b["lower"] - 1e-9 <= p <= b["upper"] + 1e-9
    assert b["lower"] - 1e-9 <= b["bma"] <= b["upper"] + 1e-9


def test_no_hidden_links_gives_point_interval_equal_oracle():
    rng = np.random.default_rng(7)
    ep = sample_episode(rng, GenConfig(hide=0.0, k_roots=(2, 3), sec_per_root=(1, 3)))
    b = posterior_bounds(ep)
    assert abs(b["upper"] - b["lower"]) < 1e-12
    assert abs(b["lower"] - oracle(ep)) < 1e-12
