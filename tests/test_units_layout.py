import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from paef.real.generic import make_window
from paef.real.units import UnitLayout


def _win(seed, D, h):
    rng = np.random.default_rng(seed)
    z = {f"u{i}": rng.integers(0, 2, 3) for i in range(6)}
    return make_window(z, 1, D, h, rng)


def _lr():
    rng = np.random.default_rng(0); X = rng.integers(0, 5, (60, 6)); y = rng.integers(0, 2, 60)
    return LogisticRegression().fit(X, y)


def test_known_duplication_invariant():
    L, lr = UnitLayout(3), _lr()
    assert np.allclose(L.unit_features(_win(1, 0, 0.0)), L.unit_features(_win(1, 8, 0.0)))


def test_interval_exact_and_midpoint():
    L, lr = UnitLayout(3), _lr()
    w = _win(3, 2, 0.6)
    hid = [c for c in w.copies if c[2]]
    base = L.unit_features(w); coef = lr.coef_.ravel(); b0 = lr.intercept_[0]
    vals = []
    for eps in itertools.product((0, 1), repeat=len(hid)):
        f = base.copy()
        for e, (_, k, _) in zip(eps, hid):
            f[2 * k + 1] += e
        vals.append(b0 + f @ coef)
    p, lo, hi = L.lineage_stacking(w, lr)
    s = lambda x: 1 / (1 + np.exp(-x))
    assert np.isclose(lo, s(min(vals))) and np.isclose(hi, s(max(vals)))
    assert np.isclose(np.log(p / (1 - p)), (min(vals) + max(vals)) / 2)


def test_presence_is_duplication_invariant():
    L = UnitLayout(3)
    assert np.allclose(L.presence_features(_win(5, 1, 0.5)), L.presence_features(_win(5, 16, 0.5)))
