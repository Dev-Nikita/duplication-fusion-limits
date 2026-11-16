"""Extended finite-cap games (manuscript: firing-set-aware mechanisms; several report types).

Unified 'signal' game. The mechanism observes a signal g (not necessarily the label) and picks the count vector a in
the admissible set B(g). With cells (B_g, p0g, p1g), p_yg = P(y, signal = g):
    R = min_x  sum_g  max_{a in B_g} [ p0g x_a^2 + p1g (1 - x_a)^2 ].
  * label-aware mechanism Q(a | u, y): signal = (u, y)  -> cells (I_D(u), p0u, 0) and (I_D(u), 0, p1u)   [Theorem 2]
  * count-only mechanism  Q(a | u)   : signal = u       -> cells (I_D(u), p0u, p1u)
  * firing-set-aware      Q(a | S)   : signal = S       -> cells (I_D(|S|), p0S, p1S)
  * K report types, coordinate-wise copying, label-aware: B = box prod_k I_D(u_k).
solve_cells returns (value, x as dict a -> value); the value is certified by evaluating the returned rule exactly.
Usage: python3 -W ignore tools/extended_models.py [--quick]   (self-checks; prints PASS/FAIL counts)"""
import itertools, os, sys
import numpy as np
import cvxpy as cp
import scipy.sparse as sp
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_cvx as C, finite_cap_solvers as S  # noqa: E402


def interval(u, D):
    return list(range(u, (D + 1) * u + 1)) if u > 0 else [0]


def box(u, D):
    return list(itertools.product(*[interval(v, D) for v in u]))


def worst_case(x, cells):
    return float(sum(max(p0 * x[a] ** 2 + p1 * (1 - x[a]) ** 2 for a in B) for B, p0, p1 in cells))


def solve_cells(cells):
    pts = sorted({a for B, _, _ in cells for a in B}); idx = {a: i for i, a in enumerate(pts)}
    x = cp.Variable(len(pts)); t = cp.Variable(len(cells)); cons = [x >= 0, x <= 1]
    pure0 = [g for g, (B, p0, p1) in enumerate(cells) if p1 == 0]; pure1 = [g for g, (B, p0, p1) in enumerate(cells) if p0 == 0 and p1 > 0]
    mixed = [g for g, (B, p0, p1) in enumerate(cells) if p0 > 0 and p1 > 0]
    obj = 0
    if pure0 or pure1:      # label-aware cells: r_g >= x_a (y=0) or s_g <= x_a (y=1), sparse incidence
        r = cp.Variable(len(cells))
        for sign, lst in ((1, pure0), (-1, pure1)):
            rows, cols = [], []
            for g in lst:
                for a in cells[g][0]:
                    rows.append(g); cols.append(idx[a])
            if rows:
                E = sp.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), cols)), shape=(len(rows), len(pts)))
                G = sp.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), rows)), shape=(len(rows), len(cells)))
                cons.append(sign * (G @ r - E @ x) >= 0)
        w0 = np.array([cells[g][1] if g in set(pure0) else 0.0 for g in range(len(cells))])
        w1 = np.array([cells[g][2] if g in set(pure1) else 0.0 for g in range(len(cells))])
        cons += [r >= 0, r <= 1]; obj = obj + w0 @ cp.square(r) + w1 @ cp.square(1 - r)
    for g in mixed:
        B, p0, p1 = cells[g]; ii = [idx[a] for a in B]
        cons.append(t[g] >= p0 * cp.square(x[ii]) + p1 * cp.square(1 - x[ii])); obj = obj + t[g]
    pr = cp.Problem(cp.Minimize(obj), cons); pr.solve(solver=cp.CLARABEL)
    xv = np.clip(x.value, 0, 1); xd = {a: float(xv[i]) for a, i in idx.items()}
    return worst_case(xd, cells), xd          # certified upper bound = exact worst case of the returned rule


def cells_label_aware(p, D):
    """p: dict u(tuple) -> (p0, p1)."""
    out = []
    for u, (p0, p1) in p.items():
        B = box(u, D)
        if p0 > 0:
            out.append((B, p0, 0.0))
        if p1 > 0:
            out.append((B, 0.0, p1))
    return out


def cells_signal(pg, D):
    """pg: dict signal -> (u(tuple), p0, p1): the mechanism sees the signal but not the label."""
    return [(box(u, D), p0, p1) for u, p0, p1 in pg.values() if p0 + p1 > 0]


def bayes(p):
    return float(sum(p0 * p1 / (p0 + p1) for p0, p1 in p if p0 + p1 > 0))


def delta_pairs(mu0, mu1, D):
    """Unmatched mass for laws on count vectors: mu_y dict u -> prob (normalised). Compatible = boxes intersect."""
    k0, k1 = list(mu0), list(mu1)
    comp = [(i, j) for i, a in enumerate(k0) for j, b in enumerate(k1)
            if all((x == 0 and z == 0) or (x > 0 and z > 0 and x <= (D + 1) * z and z <= (D + 1) * x) for x, z in zip(a, b))]
    if not comp:
        return 1.0
    n = len(comp); A = sp.lil_matrix((len(k0) + len(k1), n))
    for c, (i, j) in enumerate(comp):
        A[i, c] = 1; A[len(k0) + j, c] = 1
    res = linprog(-np.ones(n), A_ub=A.tocsr(), b_ub=np.r_[[mu0[k] for k in k0], [mu1[k] for k in k1]], bounds=[(0, None)] * n, method="highs")
    return max(0.0, 1.0 + res.fun)


def multitype_bounds(p, D):
    """p: dict u(tuple) -> (p0, p1). Returns (R_B(presence pattern), lower, upper) with
    lower = sum_pi h kappa delta^2 <= V_D <= upper = sum_pi h delta, per presence pattern pi."""
    pats = {}
    for u, (p0, p1) in p.items():
        pats.setdefault(tuple(int(v > 0) for v in u), {})[u] = (p0, p1)
    rbp = lo = up = 0.0
    for pat, cellsp in pats.items():
        a0 = sum(v[0] for v in cellsp.values()); a1 = sum(v[1] for v in cellsp.values())
        if a0 == 0 or a1 == 0:
            continue
        h = a0 * a1 / (a0 + a1); kap = 4 * a0 * a1 / (a0 + a1) ** 2; rbp += h
        if not any(pat):
            d = 0.0
        else:
            d = delta_pairs({u: v[0] / a0 for u, v in cellsp.items() if v[0] > 0}, {u: v[1] / a1 for u, v in cellsp.items() if v[1] > 0}, D)
        lo += h * kap * d * d; up += h * d
    return rbp, lo, up


if __name__ == "__main__":
    rng = np.random.default_rng(20261115); n = fail = 0; quick = "--quick" in sys.argv
    def chk(ok, msg):
        global n, fail
        n += 1
        if not ok:
            fail += 1; print("FAIL", msg)
    # 1. the unified solver reproduces the one-type label-aware and count-only solvers
    for t in range(8 if quick else 40):
        M = int(rng.integers(2, 6)); p = rng.dirichlet(np.ones(2 * (M + 1))).reshape(2, M + 1)
        for D in range(M):
            pd_ = {(u,): (p[0, u], p[1, u]) for u in range(M + 1)}
            v1, _ = solve_cells(cells_label_aware(pd_, D)); v2, _ = C.ld_minimax(p, D)
            v3, _ = solve_cells(cells_signal({u: ((u,), p[0, u], p[1, u]) for u in range(M + 1)}, D)); v4, _ = C.li_minimax(p, D)
            chk(abs(v1 - v2) < 2e-5, f"LD {v1} {v2}"); chk(abs(v3 - v4) < 2e-5, f"LI {v3} {v4}")
    # 2. firing-set-aware sandwich and the two equality cases, exhaustive laws over subsets of M sources
    for t in range(10 if quick else 60):
        M = int(rng.integers(2, 5)); subsets = [s for r in range(M + 1) for s in itertools.combinations(range(M), r)]
        mode = t % 3
        if mode == 0:        # generic law over (y, S)
            w = rng.dirichlet(np.ones(2 * len(subsets)) * 0.7).reshape(2, -1)
        elif mode == 1:      # exchangeable: P(S | y) depends on |S| only  => y independent of S given U
            q = rng.dirichlet(np.ones(2 * (M + 1))).reshape(2, M + 1)
            w = np.array([[q[y, len(s)] / sum(1 for z in subsets if len(z) == len(s)) for s in subsets] for y in (0, 1)])
        else:                # identities reveal the label: every S occurs under one label only
            lab = rng.integers(0, 2, len(subsets)); m = rng.dirichlet(np.ones(len(subsets)))
            w = np.array([[m[i] * (lab[i] == y) for i in range(len(subsets))] for y in (0, 1)])
        pu = {}
        for i, s in enumerate(subsets):
            a, b = pu.get((len(s),), (0.0, 0.0)); pu[(len(s),)] = (a + w[0, i], b + w[1, i])
        for D in range(1, M):
            vS, _ = solve_cells(cells_signal({s: ((len(s),), w[0, i], w[1, i]) for i, s in enumerate(subsets)}, D))
            vC, _ = solve_cells(cells_signal({u: (u, a, b) for u, (a, b) in pu.items()}, D)); vL, _ = solve_cells(cells_label_aware(pu, D))
            chk(vC - 2e-5 <= vS <= vL + 2e-5, f"sandwich {vC} {vS} {vL}")
            if mode == 1:
                chk(abs(vS - vC) < 3e-5, f"exchangeable {vS} {vC}")
            if mode == 2:
                chk(abs(vS - vL) < 3e-5, f"revealing {vS} {vL}")
    # 3. strict example (M = 3 sources, D = 1): R_B(U) < count-only < firing-set-aware < label-aware < R_B(P)
    sig = [((0,), 1), ((1,), 1), ((0, 1), 2), ((0, 2), 2), ((0, 1, 2), 3)]
    w = np.array([[0.30, 0.10, 0.10, 0.05, 0.00], [0.00, 0.05, 0.15, 0.00, 0.25]])
    wS = {s_: ((u,), w[0, i], w[1, i]) for i, (s_, u) in enumerate(sig)}; pu = {}
    for i, (s_, u) in enumerate(sig):
        a_, b_ = pu.get((u,), (0.0, 0.0)); pu[(u,)] = (a_ + w[0, i], b_ + w[1, i])
    vS, _ = solve_cells(cells_signal(wS, 1)); vC, _ = solve_cells(cells_signal({u: (u, a_, b_) for u, (a_, b_) in pu.items()}, 1))
    vL, _ = solve_cells(cells_label_aware(pu, 1)); rbu = bayes(pu.values()); rbp = w[0].sum() * w[1].sum() / w.sum()
    print(f"strict example (M=3, D=1): R_B(U) {rbu:.4f} < count-only {vC:.4f} < firing-set-aware {vS:.4f} < label-aware {vL:.4f} < R_B(P) {rbp:.4f}")
    chk(rbu < vC - 1e-3 < vS - 2e-3 < vL - 3e-3 < rbp - 4e-3, "strict example")
    # 4. two report types: marginals worthless, joint counts informative under D = 1, saturated at D = 2
    p2 = {(1, 3): (0.25, 0.0), (3, 1): (0.25, 0.0), (1, 1): (0.0, 0.25), (3, 3): (0.0, 0.25)}
    for D, want in ((0, 0.0), (1, 0.0), (2, 0.25)):
        v, _ = solve_cells(cells_label_aware(p2, D)); rbp, lo, up = multitype_bounds(p2, D)
        print(f"two-type example D={D}: R*_D {v:.4f} (R_B(P) {rbp:.4f}); bounds on V_D [{lo:.4f}, {up:.4f}]")
        chk(abs(v - want) < 1e-4 and lo - 1e-6 <= rbp - v <= up + 1e-6, "two-type example")
    for k in (0, 1):   # each marginal count alone carries nothing, already without copies
        m = {}
        for u, (a, b) in p2.items():
            x, z = m.get((u[k],), (0.0, 0.0)); m[(u[k],)] = (x + a, z + b)
        chk(abs(bayes(m.values()) - 0.25) < 1e-12, "marginal worthless")
    # 5. two-type bounds on random laws (K = 2, M <= 3)
    for t in range(6 if quick else 40):
        M = int(rng.integers(2, 4)); us = list(itertools.product(range(M + 1), repeat=2))
        w = rng.dirichlet(np.ones(2 * len(us)) * 0.5).reshape(2, -1) * (rng.random((2, len(us))) < 0.6); w /= w.sum()
        p = {u: (w[0, i], w[1, i]) for i, u in enumerate(us) if w[0, i] + w[1, i] > 0}
        for D in range(0, M):
            v, _ = solve_cells(cells_label_aware(p, D)); rbp, lo, up = multitype_bounds(p, D)
            chk(lo - 2e-5 <= rbp - v <= up + 2e-5, f"K=2 bounds {lo} {rbp - v} {up}")
    print(f"{n} checks, {fail} FAIL")
