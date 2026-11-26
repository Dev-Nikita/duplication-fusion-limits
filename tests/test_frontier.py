"""Remark 3 (restricted Bayes between a reference mechanism and the worst case): value concave and nondecreasing,
endpoints, and the two objective-comparison inequalities, on random laws with reference mechanism = no duplication."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
cp = pytest.importorskip("cvxpy")
import finite_cap_cvx as C, frontier as FR  # noqa: E402


@pytest.mark.parametrize("seed", range(6))
def test_restricted_bayes_frontier(seed):
    rng = np.random.default_rng(seed); M = int(rng.integers(3, 7)); D = int(rng.integers(1, 4))
    p = rng.dirichlet(np.ones(2 * (M + 1))).reshape(2, M + 1); u = np.arange(M + 1)
    ref = lambda x: float(p[0] @ x[u] ** 2 + p[1] @ (1 - x[u]) ** 2)
    eps = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]; X = [FR.restricted(p, p, D, e) for e in eps]
    R = np.array([ref(x) for x in X]); W = np.array([FR.worst(x, p, D) for x in X]); V = (1 - np.array(eps)) * R + np.array(eps) * W
    rstar = C.ld_minimax(p, D)[0]; bayes = float(np.sum(p[0] * p[1] / np.maximum(p.sum(0), 1e-300))); tol = 2e-5
    assert abs(V[0] - bayes) < tol and abs(V[-1] - rstar) < tol
    assert np.all(np.diff(V) > -tol)                                            # nondecreasing
    for i in range(1, len(eps) - 1):                                            # concave on the grid
        lam = (eps[i + 1] - eps[i]) / (eps[i + 1] - eps[i - 1])
        assert V[i] >= lam * V[i - 1] + (1 - lam) * V[i + 1] - tol
    for i in range(1, len(eps) - 1):
        e = eps[i]
        assert R[i] - R[0] <= e / (1 - e) * (W[0] - W[i]) + tol
        assert W[i] - rstar <= (1 - e) / e * (R[-1] - R[0]) + tol
