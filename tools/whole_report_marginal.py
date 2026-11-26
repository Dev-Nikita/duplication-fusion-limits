"""Whole-report copying observed through marginal counts, two report types (Proposition 3, marginal counts).
State: numbers of firing sources by pattern, n = (n10, n01, n11), n10 + n01 + n11 <= M. Observation A = (A1, A2),
A1 = n10 + n11 + t10 + t11, A2 = n01 + n11 + t01 + t11, with t_v >= 0 copies of pattern v (t_v = 0 if n_v = 0).
  reach_sources(n, D)   : reachable A by enumerating the copies of every single source (cap D per report)
  reach_formula(n, D)   : reachable A from the formula 0 <= t_v <= D n_v
  generic_value(p, N)   : minimax Brier risk, generic program over all observations in [0, N]^2 (label-aware mechanism)
  reduced_value(p)      : the three-variable program of Proposition 3(ii)
p maps (y, n) -> probability. Usage: python3 -W ignore tools/whole_report_marginal.py   (prints the checks)"""
import itertools
import numpy as np, cvxpy as cp


def states(M):
    return [n for n in itertools.product(range(M + 1), repeat=3) if sum(n) <= M]


def reach_sources(n, D):
    pats = [(1, 0)] * n[0] + [(0, 1)] * n[1] + [(1, 1)] * n[2]; out = set()
    for c in itertools.product(range(D + 1), repeat=len(pats)):
        out.add((sum(v[0] * (1 + k) for v, k in zip(pats, c)), sum(v[1] * (1 + k) for v, k in zip(pats, c))))
    return out


def reach_formula(n, D):
    return {(n[0] + n[2] + a + c, n[1] + n[2] + b + c) for a in range(D * n[0] + 1) for b in range(D * n[1] + 1) for c in range(D * n[2] + 1)}


def reach_unrestricted(n, N):
    """Reachable observations inside [0, N]^2 when the number of copies is unrestricted."""
    return {(n[0] + n[2] + a + c, n[1] + n[2] + b + c) for a in (range(N + 1) if n[0] else [0]) for b in (range(N + 1) if n[1] else [0])
            for c in (range(N + 1) if n[2] else [0]) if n[0] + n[2] + a + c <= N and n[1] + n[2] + b + c <= N}


def tau(a):
    return "0" if a == (0, 0) else "1" if a[1] == 0 else "2" if a[0] == 0 else "=" if a[0] == a[1] else ">" if a[0] > a[1] else "<"


def generic_value(p, N):
    obs = [(i, j) for i in range(N + 1) for j in range(N + 1)]; ix = {a: k for k, a in enumerate(obs)}; x = cp.Variable(len(obs)); cons = [x >= 0, x <= 1]; terms = []
    for n in {n for (_, n) in p}:
        I = [ix[a] for a in reach_unrestricted(n, N)]; r = cp.Variable(); s = cp.Variable(); cons += [r >= x[I], s <= x[I], r <= 1, s >= 0]
        terms.append(p.get((0, n), 0) * cp.square(r) + p.get((1, n), 0) * cp.square(1 - s))
    pr = cp.Problem(cp.Minimize(sum(terms)), cons); pr.solve(solver=cp.CLARABEL); return float(pr.value)


def reduced_value(p):
    """Exact cells 0, 1, 2; regions '=', '>', '<'; states with both exclusive patterns are free to choose the region."""
    cell = {k: np.zeros(2) for k in "012=><W"}
    for (y, n), w in p.items():
        s = (n[0] > 0, n[1] > 0, n[2] > 0)
        k = "0" if s == (0, 0, 0) else "1" if s == (1, 0, 0) else "2" if s == (0, 1, 0) else "=" if s == (0, 0, 1) else ">" if s == (1, 0, 1) else "<" if s == (0, 1, 1) else "W"
        cell[k][y] += w
    exact = sum(c[0] * c[1] / c.sum() for k, c in cell.items() if k in "012" and c.sum() > 0)
    x = cp.Variable(3); r = cp.Variable(); s = cp.Variable(); cons = [x >= 0, x <= 1, r >= x, s <= x, r <= 1, s >= 0]
    obj = sum(cell[k][0] * cp.square(x[i]) + cell[k][1] * cp.square(1 - x[i]) for i, k in enumerate("=><")) + cell["W"][0] * cp.square(r) + cell["W"][1] * cp.square(1 - s)
    pr = cp.Problem(cp.Minimize(obj), cons); pr.solve(solver=cp.CLARABEL); return exact + float(pr.value)


def bayes(p, stat):
    c = {}
    for (y, n), w in p.items():
        c.setdefault(stat(n), np.zeros(2))[y] += w
    return float(sum(v[0] * v[1] / v.sum() for v in c.values() if v.sum() > 0))


H = lambda n: (n[0] > 0, n[1] > 0, n[2] > 0)                     # pattern presence (Proposition 3)
P = lambda n: (n[0] + n[2] > 0, n[1] + n[2] > 0)                 # coordinate presence (Theorem 1)

if __name__ == "__main__":
    bad = 0; k = 0
    for M in (1, 2, 3):
        for n in states(M):
            for D in (0, 1, 2):
                k += 1; bad += reach_sources(n, D) != reach_formula(n, D)
            k += 1; bad += {tau(a) for a in reach_unrestricted(n, 3 * M)} != ({"0"} if H(n) == (0, 0, 0) else {"1"} if H(n) == (1, 0, 0) else {"2"} if H(n) == (0, 1, 0) else {"="} if H(n) == (0, 0, 1) else {">"} if H(n) == (1, 0, 1) else {"<"} if H(n) == (0, 1, 1) else {"=", ">", "<"})
    rng = np.random.default_rng(20261116); gap = 0; strict_lo = strict_hi = 0
    for trial in range(40):
        M = int(rng.integers(1, 4)); S = states(M); w = rng.dirichlet(np.ones(2 * len(S)) * rng.choice([0.3, 1.0]))
        p = {(y, n): float(w[y * len(S) + i]) for y in (0, 1) for i, n in enumerate(S)}
        g = generic_value(p, 2 * M); r = reduced_value(p); lo = bayes(p, H); hi = bayes(p, P); k += 3
        gap = max(gap, abs(g - r)); bad += abs(g - r) > 1e-5; bad += lo > r + 1e-6; bad += r > hi + 1e-6; strict_lo += r > lo + 1e-4; strict_hi += hi > r + 1e-4
    # the example of the text: (1,1),(1,0) under y=0 versus (1,1),(0,1) under y=1
    ex = {(0, (1, 0, 1)): 0.5, (1, (0, 1, 1)): 0.5}; k += 2; bad += abs(reduced_value(ex)) > 1e-6; bad += abs(bayes(ex, P) - 0.25) > 1e-9
    print(f"whole-report / marginal counts: {k} checks, {bad} failures; max |generic - reduced| = {gap:.1e}; strictly above R_B(H) in {strict_lo}/40, strictly below R_B(P) in {strict_hi}/40")
