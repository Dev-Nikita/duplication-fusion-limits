import numpy as np
from paef.real.mechanisms_r10 import make_window_r10


def _z():
    rng = np.random.default_rng(1); return {f"u{i}": rng.integers(0, 2, 4) for i in range(20)}


def _storm_rate(mech, y, n=2000):
    rng = np.random.default_rng(0); z = _z()
    return np.mean([len(make_window_r10(z, y, mech, 0.0, rng).copies) > 0 for _ in range(n)])


def test_ld_pair_is_mirror():
    assert abs(_storm_rate("storm_inc", 1) - 0.8) < 0.03 and abs(_storm_rate("storm_inc", 0) - 0.2) < 0.03
    assert abs(_storm_rate("storm_ben", 1) - 0.2) < 0.03 and abs(_storm_rate("storm_ben", 0) - 0.8) < 0.03


def test_li_storm_label_independent():
    assert abs(_storm_rate("li_storm", 1) - _storm_rate("li_storm", 0)) < 0.04


def test_copies_only_of_firing():
    z = _z(); w = make_window_r10(z, 1, "li_burst", 0.5, np.random.default_rng(3))
    assert all(z[o][k] == 1 for o, k, _ in w.copies)
