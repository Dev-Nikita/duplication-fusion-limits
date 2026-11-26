"""Exact rational brackets for Proposition 5 and for the marginal-count part of Proposition 3.
A float solver proposes a rule and a mechanism; both are rounded to rationals and evaluated exactly:
  upper bound of the minimax value = exact worst case of the rational rule,
  lower bound = exact Bayes risk under the rational mechanism.
The claims are then tested against these exact bounds. A pass certifies the inequalities up to the bracket width;
it is not a proof. Run as a script to print the summary:  python3 -W ignore tests/r4/test_exact_brackets.py"""
import os, sys
from fractions import Fraction as Fr
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cp = pytest.importorskip("cvxpy")
import finite_cap_cvx as C, finite_cap_solvers as S, whole_report_marginal as W  # noqa: E402
from test_exact_small_cases import maxflow  # noqa: E402

rat = lambda v: Fr(float(v)).limit_denominator(10 ** 7)


def bayes(w0, w1):
    return sum((a * b / (a + b) for a, b in zip(w0, w1) if a + b > 0), Fr(0))


def prop5_case(rng):
    M = int(rng.integers(2, 5)); D = int(rng.integers(0, 3)); den = 12
    c = rng.multinomial(den * 2, rng.dirichlet(np.ones(2 * (M + 1)) * 0.7)).reshape(2, M + 1)
    if c[0, 1:].sum() == 0 or c[1, 1:].sum() == 0:
        return None
    P = [[Fr(int(k), 2 * den) for k in row] for row in c]; p = np.array(c, float) / c.sum(); I = [list(i) for i in S.intervals(M, D)]
    x = [min(max(rat(v), Fr(0)), Fr(1)) for v in C.ld_minimax(p, D)[1]]
    up = sum(P[0][u] * max(x[a] ** 2 for a in I[u]) + P[1][u] * max((1 - x[a]) ** 2 for a in I[u]) for u in range(M + 1))
    Q = S.solve_dual(p, D, iters=4000)[3]; n = (D + 1) * M + 1; w = [[Fr(0)] * n, [Fr(0)] * n]
    for y in (0, 1):
        for u in range(M + 1):
            q = [max(rat(v), Fr(0)) for v in Q[y][u]]; t = sum(q); q = [v / t for v in q]
            for a, v in zip(I[u], q):
                w[y][a] += P[y][u] * v
    lo = bayes(w[0], w[1]); assert lo <= up
    pi = [sum(P[0][1:]), sum(P[1][1:])]; mu = [[v / pi[y] for v in P[y][1:]] for y in (0, 1)]
    delta = 1 - maxflow(mu[0], mu[1], D); h = pi[0] * pi[1] / (pi[0] + pi[1]); kappa = 4 * pi[0] * pi[1] / (pi[0] + pi[1]) ** 2
    rbp = h + (P[0][0] * P[1][0] / (P[0][0] + P[1][0]) if P[0][0] + P[1][0] > 0 else 0)
    v_lo, v_hi = rbp - up, rbp - lo                      # exact bracket of V_D
    return dict(ok=(h * kappa * delta ** 2 <= v_hi) and (v_lo <= h * delta), cert_lower=h * kappa * delta ** 2 <= v_lo,
                cert_upper=v_hi <= h * delta, width=float(up - lo), sat=(delta == 0), v_hi=float(v_hi))


def marginal_case(rng):
    M = int(rng.integers(1, 3)); St = W.states(M); den = 24
    c = rng.multinomial(den, rng.dirichlet(np.ones(2 * len(St)) * 0.6)).reshape(2, len(St))
    P = {(y, n): Fr(int(c[y, i]), den) for y in (0, 1) for i, n in enumerate(St) if c[y, i] > 0}
    kind = lambda n: {(0, 0, 0): "0", (1, 0, 0): "1", (0, 1, 0): "2", (0, 0, 1): "=", (1, 0, 1): ">", (0, 1, 1): "<"}.get(tuple(int(v > 0) for v in n), "W")
    cell = {k: [Fr(0), Fr(0)] for k in "012=><W"}
    for (y, n), v in P.items():
        cell[kind(n)][y] += v
    # float solution of the three-variable program, rounded
    x = cp.Variable(3); r = cp.Variable(); s = cp.Variable(); f = lambda k, y: float(cell[k][y])
    obj = sum(f(k, 0) * cp.square(x[i]) + f(k, 1) * cp.square(1 - x[i]) for i, k in enumerate("=><")) + f("W", 0) * cp.square(r) + f("W", 1) * cp.square(1 - s)
    cp.Problem(cp.Minimize(obj), [x >= 0, x <= 1, r >= x, s <= x, r <= 1, s >= 0]).solve(solver=cp.CLARABEL)
    xr = {k: min(max(rat(v), Fr(0)), Fr(1)) for k, v in zip("=><", x.value)}
    for k in "012":
        xr[k] = cell[k][1] / (cell[k][0] + cell[k][1]) if cell[k][0] + cell[k][1] > 0 else Fr(1, 2)
    # exact worst case of the tau-rule in the generic game on the grid [0, 2M]^2
    up = Fr(0)
    for (y, n), v in P.items():
        vals = [xr[W.tau(a)] for a in W.reach_unrestricted(n, 2 * M)]
        up += v * (max(t ** 2 for t in vals) if y == 0 else max((1 - t) ** 2 for t in vals))
    # exact Bayes risk under a canonical-point mechanism: free states of each class are split over the three points
    l0 = cp.Variable(3, nonneg=True); l1 = cp.Variable(3, nonneg=True); terms = []
    for i, k in enumerate("=><"):
        u0 = f(k, 0) + f("W", 0) * l0[i]; u1 = f(k, 1) + f("W", 1) * l1[i]
        terms.append(u0 - cp.quad_over_lin(u0, u0 + u1 + 1e-12))
    cp.Problem(cp.Maximize(sum(terms)), [cp.sum(l0) == 1, cp.sum(l1) == 1]).solve(solver=cp.CLARABEL)
    def simplex(v):
        q = [max(rat(t), Fr(0)) for t in v]; t = sum(q); return [t_ / t for t_ in q]
    L0, L1 = simplex(l0.value), simplex(l1.value); obs = {k: [Fr(0), Fr(0)] for k in "012=><"}
    for k in "012=><":
        obs[k][0] += cell[k][0]; obs[k][1] += cell[k][1]
    for i, k in enumerate("=><"):
        obs[k][0] += cell["W"][0] * L0[i]; obs[k][1] += cell["W"][1] * L1[i]
    lo = bayes([obs[k][0] for k in "012=><"], [obs[k][1] for k in "012=><"])
    bh = bayes(*zip(*[(sum(v for (y, n), v in P.items() if y == 0 and tuple(int(t > 0) for t in n) == g), sum(v for (y, n), v in P.items() if y == 1 and tuple(int(t > 0) for t in n) == g)) for g in {tuple(int(t > 0) for t in n) for (_, n) in P}]))
    pp = lambda n: (n[0] + n[2] > 0, n[1] + n[2] > 0)
    bp = bayes(*zip(*[(sum(v for (y, n), v in P.items() if y == 0 and pp(n) == g), sum(v for (y, n), v in P.items() if y == 1 and pp(n) == g)) for g in {pp(n) for (_, n) in P}]))
    return dict(ok=(lo <= up) and (bh <= up) and (lo <= bp), width=float(up - lo), cert_sandwich=(bh <= up) and (lo <= bp))


def summary(n5=60, nm=40):
    rng = np.random.default_rng(20261116); A = [a for a in (prop5_case(rng) for _ in range(n5)) if a]; B = [marginal_case(rng) for _ in range(nm)]
    return A, B


def test_exact_brackets():
    A, B = summary(30, 20)
    assert all(a["ok"] for a in A) and all(b["ok"] for b in B)
    assert max(b["width"] for b in B) < 1e-5


if __name__ == "__main__":
    A, B = summary()
    print(f"Proposition 5: {len(A)} laws; violations beyond the bracket: {sum(not a['ok'] for a in A)}; lower bound certified exactly in {sum(a['cert_lower'] for a in A)}, "
          f"upper bound certified exactly in {sum(a['cert_upper'] for a in A)}; saturated (delta = 0) cases: {sum(a['sat'] for a in A)}; max bracket width {max(a['width'] for a in A):.1e}")
    print(f"Proposition 3 (marginal counts): {len(B)} laws; violations: {sum(not b['ok'] for b in B)}; max bracket width {max(b['width'] for b in B):.1e}; "
          f"R_B(H) <= upper and lower <= R_B(P) hold exactly in {sum(b['cert_sandwich'] for b in B)}")
