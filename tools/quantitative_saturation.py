"""Quantitative saturation (Proposition 4 of the manuscript), verified numerically.

Setting: one report type, p[y,u] = P(y, U=u); pi_y = P(y, U>0); mu_y = law of U given (y, U>0).
Compatible pairs (u, u'): u <= (D+1) u' and u' <= (D+1) u (the intervals I_D(u), I_D(u') intersect).
  m_D     = max mass of a sub-coupling of (mu_0, mu_1) supported on compatible pairs (an LP)
  delta_D = 1 - m_D  (= min TV between the laws of A given (y, U>0) reachable in the two classes)
  beta_D  = Lebesgue measure of {s in (0,1): max(q1(s)/q0(s), q0(s)/q1(s)) > D+1}  (comonotone mismatch)
  V_D     = R_B(P) - R*_D  (what counts are still worth beyond presence under cap D)
Claim:  h * kappa * delta_D^2 <= V_D <= h * delta_D <= h * beta_D,
        h = pi0 pi1 / (pi0 + pi1) (presence Bayes risk on U>0),  kappa = 4 pi0 pi1 / (pi0 + pi1)^2 in (0, 1].
Usage: python3 tools/quantitative_saturation.py [--quick]"""
import os, sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_solvers as S  # noqa: E402


def cond(p):
    pi = p[:, 1:].sum(1); return p[0, 1:] / pi[0], p[1, 1:] / pi[1], pi


def delta(p, D):
    mu0, mu1, _ = cond(p); M = len(mu0); u = np.arange(1, M + 1)
    comp = (u[:, None] <= (D + 1) * u[None, :]) & (u[None, :] <= (D + 1) * u[:, None])
    idx = np.argwhere(comp); n = len(idx)
    Aub = np.zeros((2 * M, n)); Aub[idx[:, 0], np.arange(n)] = 1; Aub[M + idx[:, 1], np.arange(n)] = 1
    res = linprog(-np.ones(n), A_ub=Aub, b_ub=np.r_[mu0, mu1], bounds=[(0, None)] * n, method="highs")
    return max(0.0, 1.0 + res.fun)


def beta(p, D):
    mu0, mu1, _ = cond(p); F0, F1 = np.cumsum(mu0), np.cumsum(mu1)
    br = np.unique(np.r_[0.0, F0, F1, 1.0]); br = br[(br >= 0) & (br <= 1)]; tot = 0.0
    for a, b in zip(br[:-1], br[1:]):
        if b - a <= 1e-15:
            continue
        s = 0.5 * (a + b); q0 = int(np.searchsorted(F0, s - 1e-12) + 1); q1 = int(np.searchsorted(F1, s - 1e-12) + 1)
        if max(q0 / q1, q1 / q0) > D + 1 + 1e-12:
            tot += b - a
    return tot


def constants(p):
    _, _, pi = cond(p); h = pi[0] * pi[1] / pi.sum(); return h * 4 * pi[0] * pi[1] / pi.sum() ** 2, h


if __name__ == "__main__":
    import cvxpy  # noqa: F401  (interior-point value of R*_D)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import finite_cap_cvx as C
    rng = np.random.default_rng(20261027); n = fail = 0; worst = []
    for trial in range(40 if "--quick" in sys.argv else 400):
        M = int(rng.integers(2, 13)); p = np.zeros((2, M + 1))
        for y in (0, 1):
            k = int(rng.integers(1, M + 1)); sup = rng.choice(np.arange(1, M + 1), size=k, replace=False)
            p[y, sup] = rng.dirichlet(np.ones(k) * rng.choice([0.3, 1, 3])); p[y, 0] = rng.random() * rng.choice([0, 1])
        p /= p.sum(); c1, c2 = constants(p); rbp = S.bayes_risk_presence(p)
        for D in range(0, M):
            v, x = C.ld_minimax(p, D); up = S.primal_value(x, p, D)   # certified upper bound on R*_D
            V_lo, V_hi = rbp - up, rbp - v                             # bracket for V_D (solver tolerance)
            d, b = delta(p, D), beta(p, D)
            ok = (c1 * d ** 2 <= V_hi + 1e-6) and (V_lo <= c2 * d + 1e-6) and (d <= b + 1e-9) and ((d < 1e-9) == (V_hi < 1e-6) or d > 1e-6)
            n += 1; fail += (not ok)
            if not ok:
                print("FAIL", M, D, d, b, V_lo, V_hi, c1, c2)
            if d > 1e-9:
                worst.append(((V_lo) / (c2 * d), V_hi / (c1 * d * d)))
    # sharpness examples of the proof (D = 0, equal priors) and the Lipschitz property
    for t in (0.0, 0.2, 0.5, 0.8, 1.0):
        for kind in ("upper", "lower"):
            p = np.zeros((2, 4))
            if kind == "upper":
                p[0, 2] = p[1, 2] = (1 - t) / 2; p[0, 1] = p[1, 3] = t / 2
            else:
                p[0, 1] = p[1, 2] = (1 + t) / 4; p[0, 2] = p[1, 1] = (1 - t) / 4
            c1, h = constants(p); V = S.bayes_risk_presence(p) - S.bayes_risk_U(p); d = delta(p, 0)
            ok = abs(d - t) < 1e-9 and abs(V - (h * d if kind == "upper" else c1 * d * d)) < 1e-9
            n += 1; fail += (not ok)
            if not ok:
                print("FAIL sharpness", kind, t, d, V)
    for trial in range(200):
        M = int(rng.integers(2, 10)); D = int(rng.integers(0, M))
        mu = rng.dirichlet(np.ones(M), size=4)
        def dl(m0, m1):
            p = np.zeros((2, M + 1)); p[0, 1:] = m0 / 2; p[1, 1:] = m1 / 2; return delta(p, D)
        lhs = abs(dl(mu[0], mu[1]) - dl(mu[2], mu[3])); rhs = 0.5 * np.abs(mu[0] - mu[2]).sum() + 0.5 * np.abs(mu[1] - mu[3]).sum()
        n += 1; fail += (lhs > rhs + 1e-9)
    w = np.array(worst)
    print(f"{n} checks, {fail} FAIL; tightness: V/(h*delta) max {w[:, 0].max():.4f}, median {np.median(w[:, 0]):.3f}; "
          f"V/(h*kappa*delta^2) min {w[:, 1].min():.4f}")
