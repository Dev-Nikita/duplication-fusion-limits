"""Independent solvers for the finite-cap minimax game (Theorem 2 of the manuscript), written without reference to
tools/verify_finite_cap_game.py.

Game (one report type, Brier loss): p[y, u] = P(Y = y, U = u), u = 0..M; admissible counts I_D(u) = {u..(D+1)u},
I_D(0) = {0}. The fuser picks x_a in [0, 1] for every count a; the adversary picks a kernel Q(a | u, y) on I_D(u).

  primal_value(x)  = sum_u p0u * max_{a in I(u)} x_a^2 + p1u * max_{a in I(u)} (1 - x_a)^2
                     (exact worst-case risk of the fixed rule x; any x gives an UPPER bound on R*_D)
  dual_value(Q)    = sum_a w0a * w1a / (w0a + w1a),  w_ya = sum_u p_yu Q(a | u, y)
                     (Bayes risk of the channel Q; any admissible Q gives a LOWER bound on R*_D)

Solver A: the convex QP of the manuscript, solved with scipy (SLSQP), returns x_A and primal_value(x_A).
Solver B: exponentiated-gradient (mirror) ascent on the concave dual, numpy only; it returns Q_B, the lower bound
          dual_value(Q_B) and the upper bound primal_value(x(Q_B)) of the induced Bayes rule. The gap between the
          two certifies optimality independently of solver A."""
import numpy as np


def intervals(M, D):
    return [np.arange(u, (D + 1) * u + 1) if u > 0 else np.array([0]) for u in range(M + 1)]


def primal_value(x, p, D):
    M = p.shape[1] - 1; I = intervals(M, D)
    return float(sum(p[0, u] * np.max(x[I[u]] ** 2) + p[1, u] * np.max((1 - x[I[u]]) ** 2) for u in range(M + 1)))


def dual_value(Q0, Q1, p, D):
    M = p.shape[1] - 1; I = intervals(M, D); A = (D + 1) * M + 1
    w0 = np.zeros(A); w1 = np.zeros(A)
    for u in range(M + 1):
        w0[I[u]] += p[0, u] * Q0[u]; w1[I[u]] += p[1, u] * Q1[u]
    s = w0 + w1; m = s > 0
    return float(np.sum(w0[m] * w1[m] / s[m])), w0, w1


def bayes_rule(w0, w1, fill=None):
    s = w0 + w1; x = np.where(s > 0, w1 / np.where(s > 0, s, 1), np.nan)
    if fill is None:
        fill = np.nanmean(x) if np.isfinite(x).any() else 0.5
    return np.where(np.isnan(x), fill, x)


def solve_dual(p, D, iters=20000, eta=4.0, tol=1e-9):
    """Solver B: mirror ascent on the concave dual. Returns (lower, upper, x, Q0, Q1)."""
    M = p.shape[1] - 1; I = intervals(M, D)
    Q0 = [np.full(len(I[u]), 1.0 / len(I[u])) for u in range(M + 1)]
    Q1 = [q.copy() for q in Q0]
    best_lo, best_up, best_x, best_Q = -1.0, 2.0, None, None
    for t in range(iters):
        lo, w0, w1 = dual_value(Q0, Q1, p, D)
        x = bayes_rule(w0, w1)
        # fill counts with no mass by the value that minimises the primal (any value works: they carry no mass now,
        # but they must not create a larger maximum; clip to the range of neighbouring values)
        up = primal_value(x, p, D)
        if lo > best_lo:
            best_lo, best_Q = lo, ([q.copy() for q in Q0], [q.copy() for q in Q1])
        if up < best_up:
            best_up, best_x = up, x.copy()
        if best_up - best_lo < tol:
            break
        step = eta / np.sqrt(t + 1)
        for u in range(M + 1):
            g0 = x[I[u]] ** 2; g1 = (1 - x[I[u]]) ** 2
            Q0[u] = Q0[u] * np.exp(step * (g0 - g0.max())); Q0[u] /= Q0[u].sum()
            Q1[u] = Q1[u] * np.exp(step * (g1 - g1.max())); Q1[u] /= Q1[u].sum()
    return best_lo, best_up, best_x, best_Q


def solve_primal(p, D, x0=None):
    """Solver A: the convex QP (variables x_a, r_u, s_u) with scipy SLSQP. Returns (value, x)."""
    from scipy.optimize import minimize
    M = p.shape[1] - 1; I = intervals(M, D); A = (D + 1) * M + 1; n = A + 2 * (M + 1)
    def f(z):
        r = z[A:A + M + 1]; s = z[A + M + 1:]
        return float(np.sum(p[0] * r ** 2 + p[1] * (1 - s) ** 2))
    def g(z):
        out = np.zeros(n); r = z[A:A + M + 1]; s = z[A + M + 1:]
        out[A:A + M + 1] = 2 * p[0] * r; out[A + M + 1:] = -2 * p[1] * (1 - s); return out
    rows = []
    for u in range(M + 1):
        for a in I[u]:
            e = np.zeros(n); e[A + u] = 1; e[a] = -1; rows.append(e)            # r_u - x_a >= 0
            e = np.zeros(n); e[a] = 1; e[A + M + 1 + u] = -1; rows.append(e)    # x_a - s_u >= 0
    G = np.array(rows)
    cons = [{"type": "ineq", "fun": lambda z: G @ z, "jac": lambda z: G}]
    z0 = np.full(n, 0.5) if x0 is None else np.r_[x0, np.full(2 * (M + 1), 0.5)]
    res = minimize(f, z0, jac=g, bounds=[(0, 1)] * n, constraints=cons, method="SLSQP",
                   options={"maxiter": 2000, "ftol": 1e-12})
    x = np.clip(res.x[:A], 0, 1)
    return primal_value(x, p, D), x


def bayes_risk_U(p):
    s = p.sum(0); m = s > 0
    return float(np.sum(p[0, m] * p[1, m] / s[m]))


def bayes_risk_presence(p):
    a0, a1 = p[0, 0], p[1, 0]; b0, b1 = p[0, 1:].sum(), p[1, 1:].sum()
    return float((a0 * a1 / (a0 + a1) if a0 + a1 > 0 else 0) + (b0 * b1 / (b0 + b1) if b0 + b1 > 0 else 0))
