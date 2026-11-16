"""A4 interval is exact: equals min/max over all {COPY, FRESH} assignments of hidden copies."""
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression

from paef.real.origami import NODES, Window, lineage_stacking, unit_features


def test_interval_equals_enumeration():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, (200, len(NODES) * 6)); y = rng.integers(0, 2, 200)
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    for _ in range(30):
        nodes = list(rng.choice(NODES, 2, replace=False))
        w = Window(y=1, z={n: rng.integers(0, 2, 3) for n in nodes})
        for n in nodes:
            for k in range(3):
                if w.z[n][k]:
                    for _ in range(int(rng.integers(0, 3))):
                        w.copies.append((n, k, bool(rng.random() < 0.6)))
        o = lineage_stacking(w, lr)
        hid = [c for c in w.copies if c[2]]
        vals = []
        for choice in itertools.product((0, 1), repeat=len(hid)):
            f = unit_features(w, "exclude")
            for (n, k, _), c in zip(hid, choice):
                if c:
                    f[(NODES.index(n) * 3 + k) * 2 + 1] += 1
            vals.append(1 / (1 + np.exp(-(lr.intercept_[0] + f @ lr.coef_.ravel()))))
        assert abs(min(vals) - o["ls_lower"]) < 1e-12 and abs(max(vals) - o["ls_upper"]) < 1e-12
