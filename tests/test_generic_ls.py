import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from paef.real.generic import Layout, Window


def test_generic_interval_exact():
    L = Layout({"A": ["a0", "a1"], "B": ["b0", "b1", "b2"]})
    rng = np.random.default_rng(1)
    X = rng.integers(0, 3, (200, len(L.slots) * 2)); y = rng.integers(0, 2, 200)
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    for _ in range(20):
        w = Window(y=1, z={"A": rng.integers(0, 2, 2), "B": rng.integers(0, 2, 3)})
        for o, zz in w.z.items():
            for k, v in enumerate(zz):
                if v:
                    for _ in range(int(rng.integers(0, 3))):
                        w.copies.append((o, k, bool(rng.random() < .6)))
        p, lo, hi = L.lineage_stacking(w, lr)
        hid = [c for c in w.copies if c[2]]; vals = []
        for ch in itertools.product((0, 1), repeat=len(hid)):
            f = L.unit_features(w)
            for (o, k, _), c in zip(hid, ch):
                f[L.index[(o, k)] * 2 + 1] += c
            vals.append(1 / (1 + np.exp(-(lr.intercept_[0] + f @ lr.coef_.ravel()))))
        assert abs(min(vals) - lo) < 1e-12 and abs(max(vals) - hi) < 1e-12
