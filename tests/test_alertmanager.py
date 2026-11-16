import numpy as np
from paef.real.alertmanager import make_window_am, count_features_am, la_features_am
from paef.real.units import UnitLayout

L = UnitLayout(4)


def _z():
    return {"a": np.array([1, 1, 0, 0]), "b": np.array([1, 0, 1, 0]), "c": np.array([0, 0, 0, 0])}


def _w(s, **kw):
    return make_window_am(_z(), 1, s, np.random.default_rng(0), **kw)


def test_default_no_copies():
    w = _w("am_default"); assert w.copies == [] and w.invisible == []


def test_inhibition_removes_latency_of_rejected_ue():
    w = _w("am_inhibit"); assert list(w.z["b"]) == [0, 0, 1, 0] and list(w.z["a"]) == [1, 1, 0, 0]


def test_replica_is_invisible_and_counted_as_unit():
    w = _w("am_ha_replica"); assert len(w.invisible) == 4 and w.copies == []
    c = count_features_am(L, w).reshape(4, 2)[:, 1]; assert list(c) == [4, 2, 2, 0]
    u = la_features_am(L, w)[:8].reshape(4, 2)[:, 1]; assert list(u) == [4, 2, 2, 0]


def test_multiroute_and_combined_copy_counts():
    assert len(_w("am_multiroute").copies) == 8
    w = _w("am_combined"); assert len(w.invisible) == 3 and len(w.copies) == 12


def test_truncation_caps_alerts_per_type():
    w = _w("am_truncate", max_alerts=1); assert all(sum(int(zz[k]) for zz in w.z.values()) <= 1 for k in range(4))
