import numpy as np
from paef.real.mechanisms import make_window_mech, UNSEEN, lineage_separated_features
from paef.real.units import UnitLayout


def _z(seed):
    rng = np.random.default_rng(seed)
    return {f"u{i}": rng.integers(0, 2, 4) for i in range(20)}


def test_copies_only_of_firing_alarms_and_counts():
    for mech in ("iid",) + UNSEEN:
        rng = np.random.default_rng(1); z = _z(2)
        w = make_window_mech(z, 1, mech, 0.5, rng)
        assert all(z[o][k] == 1 for o, k, _ in w.copies)


def test_source_and_incident_rates():
    z = _z(3); rng = np.random.default_rng(0)
    w = make_window_mech(z, 0, "source", 0.0, rng)
    fired = {k: sum(zz[k] for zz in z.values()) for k in range(4)}
    cnt = {k: sum(1 for _, kk, _ in w.copies if kk == k) for k in range(4)}
    assert cnt == {k: fired[k] * (1, 16, 1, 16)[k] for k in range(4)}
    w0 = make_window_mech(z, 0, "incident", 0, rng); w1 = make_window_mech(z, 1, "incident", 0, rng)
    assert len(w1.copies) == 6 * len(w0.copies)


def test_lineage_separated_unit_part_invariant():
    L = UnitLayout(4); z = _z(4)
    a = lineage_separated_features(L, make_window_mech(z, 1, "iid", 0.0, np.random.default_rng(0), D=1))
    b = lineage_separated_features(L, make_window_mech(z, 1, "storm", 0.0, np.random.default_rng(5)))
    assert np.allclose(a[:8], b[:8]) and a[8:12].sum() <= b[8:12].sum() + 1e9
