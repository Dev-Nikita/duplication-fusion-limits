"""Split-conformal quantile and joint coverage (T1) on exchangeable scores."""
import numpy as np

from paef.calibration.conformal_sets import conformal_quantile


def test_quantile_index():
    s = np.arange(1, 20) / 20.0  # n = 19
    assert conformal_quantile(s, 0.1) == s[17]  # ceil(20*0.9)=18 -> 18th smallest
    assert conformal_quantile(s[:5], 0.05) == 1.0  # ceil(6*0.95)=6 > 5


def test_marginal_coverage_bounds_monte_carlo():
    rng = np.random.default_rng(0)
    n, delta, reps = 99, 0.1, 4000
    hits = 0
    for _ in range(reps):
        s = rng.random(n + 1)
        hits += s[-1] <= conformal_quantile(s[:-1], delta)
    cov = hits / reps
    assert 1 - delta - 0.015 <= cov <= 1 - delta + 1 / (n + 1) + 0.015


def test_max_score_gives_joint_and_per_node_does_not():
    rng = np.random.default_rng(1)
    n, m, delta, reps = 400, 5, 0.1, 1500
    joint_max, joint_marg = 0, 0
    for _ in range(reps):
        S = rng.random((n + 1, m))
        qmax = conformal_quantile(S[:-1].max(1), delta)
        qnode = conformal_quantile(S[:-1].ravel(), delta)
        joint_max += (S[-1] <= qmax).all()
        joint_marg += (S[-1] <= qnode).all()
    assert joint_max / reps >= 1 - delta - 0.02
    assert joint_marg / reps < 1 - delta - 0.2
