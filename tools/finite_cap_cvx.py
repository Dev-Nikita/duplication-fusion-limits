"""Convex-programming solvers (cvxpy/Clarabel) for the finite-cap game and two variants, used for the
label-independent comparison and the estimation-error analysis. Cross-checked against finite_cap_solvers.py.

  ld_minimax(p, D)          : the label-dependent game of Theorem 2 (QP with interval max/min variables)
  li_minimax(p, D)          : label-independent kernels Q(a|u):  min_x sum_u max_{a in I(u)} [p0u x_a^2 + p1u (1-x_a)^2]
  tv_robust_minimax(p,D,eps): min_x  L_D(x; p) + eps * max_{y,u} c_yu(x),  c_0u = max_a x_a^2, c_1u = max_a (1-x_a)^2;
                              an upper bound on  sup_{TV(p', p) <= eps} L_D(x; p')  (moving eps of mass to the worst cell)
All return (value, x)."""
import numpy as np
import cvxpy as cp


def _intervals(M, D):
    return [np.arange(u, (D + 1) * u + 1) if u > 0 else np.array([0]) for u in range(M + 1)]


def _ld_parts(p, D):
    M = p.shape[1] - 1; I = _intervals(M, D); A = (D + 1) * M + 1
    x = cp.Variable(A); r = cp.Variable(M + 1); s = cp.Variable(M + 1)
    cons = [x >= 0, x <= 1, r <= 1, s >= 0]
    for u in range(M + 1):
        cons += [r[u] >= x[I[u]], s[u] <= x[I[u]]]
    # r_u^2 is the worst y=0 loss at u, (1-s_u)^2 the worst y=1 loss (r >= x >= 0 and s <= x <= 1 make this exact)
    c0 = cp.square(r); c1 = cp.square(1 - s)
    return x, cons, c0, c1, M


def ld_minimax(p, D):
    x, cons, c0, c1, M = _ld_parts(p, D)
    pr = cp.Problem(cp.Minimize(p[0] @ c0 + p[1] @ c1), cons); pr.solve(solver=cp.CLARABEL)
    return float(pr.value), np.clip(x.value, 0, 1)


def tv_robust_minimax(p, D, eps):
    x, cons, c0, c1, M = _ld_parts(p, D)
    t = cp.Variable()
    cons += [t >= c0, t >= c1]
    pr = cp.Problem(cp.Minimize(p[0] @ c0 + p[1] @ c1 + eps * t), cons); pr.solve(solver=cp.CLARABEL)
    return float(pr.value), np.clip(x.value, 0, 1)


def li_minimax(p, D):
    M = p.shape[1] - 1; I = _intervals(M, D); A = (D + 1) * M + 1
    x = cp.Variable(A); t = cp.Variable(M + 1); cons = [x >= 0, x <= 1]
    for u in range(M + 1):
        cons.append(t[u] >= p[0, u] * cp.square(x[I[u]]) + p[1, u] * cp.square(1 - x[I[u]]))
    pr = cp.Problem(cp.Minimize(cp.sum(t)), cons); pr.solve(solver=cp.CLARABEL)
    return float(pr.value), np.clip(x.value, 0, 1)


def li_worst_case(x, p, D):
    """Exact worst case of a fixed rule over label-independent kernels."""
    M = p.shape[1] - 1; I = _intervals(M, D)
    return float(sum(np.max(p[0, u] * x[I[u]] ** 2 + p[1, u] * (1 - x[I[u]]) ** 2) for u in range(M + 1)))
