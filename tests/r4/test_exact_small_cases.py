"""Exact rational checks of Proposition 4 on small cases, written without importing any project solver.
For all pairs of laws of U given U>0 with masses in multiples of 1/4 on {1..M} (M <= 4) and caps D = 0..3:
  (b) a coupling supported on compatible pairs exists   (exact max-flow with Fractions)
  (c) F0(t) <= F1((D+1)t) and F1(t) <= F0((D+1)t) for all integers t >= 1
  (d) rho <= D+1, rho = sup_s max(q1(s)/q0(s), q0(s)/q1(s)) with left-continuous quantiles
must agree, and D_sat = ceil(rho) - 1 <= M - 1 must be the smallest cap with (b)."""
import itertools, math
from collections import deque
from fractions import Fraction as Fr


def laws(M, den=4):
    for c in itertools.product(range(den + 1), repeat=M):
        if sum(c) == den:
            yield [Fr(k, den) for k in c]


def maxflow(mu0, mu1, D):
    """Exact max-flow from class-0 atoms to class-1 atoms over pairs whose copy intervals intersect."""
    M = len(mu0); n = 2 * M + 2; s, t = 2 * M, 2 * M + 1; cap = [[Fr(0)] * n for _ in range(n)]
    for i in range(M):
        cap[s][i] = mu0[i]; cap[M + i][t] = mu1[i]
        for j in range(M):
            u, v = i + 1, j + 1
            if max(u, v) <= (D + 1) * min(u, v):            # [u,(D+1)u] and [v,(D+1)v] intersect
                cap[i][M + j] = Fr(2)
    flow = Fr(0)
    while True:
        par = [-1] * n; par[s] = s; q = deque([s])
        while q and par[t] < 0:
            a = q.popleft()
            for b in range(n):
                if par[b] < 0 and cap[a][b] > 0:
                    par[b] = a; q.append(b)
        if par[t] < 0:
            return flow
        f = None; b = t
        while b != s:
            f = cap[par[b]][b] if f is None else min(f, cap[par[b]][b]); b = par[b]
        b = t
        while b != s:
            cap[par[b]][b] -= f; cap[b][par[b]] += f; b = par[b]
        flow += f


def cdf(mu, t):
    return sum(mu[: min(int(t), len(mu))])


def cond_c(mu0, mu1, D):
    M = len(mu0)
    return all(cdf(mu0, t) <= cdf(mu1, (D + 1) * t) and cdf(mu1, t) <= cdf(mu0, (D + 1) * t) for t in range(1, M + 1))


def rho(mu0, mu1):
    M = len(mu0); F0 = [cdf(mu0, t) for t in range(1, M + 1)]; F1 = [cdf(mu1, t) for t in range(1, M + 1)]
    pts = sorted(set(F0 + F1 + [Fr(0)])); mids = [(a + b) / 2 for a, b in zip(pts, pts[1:])] + [p for p in pts if 0 < p < 1] + [Fr(1) - Fr(1, 10 ** 6)]
    q = lambda F, s: next(u + 1 for u in range(M) if F[u] >= s)
    return max(max(Fr(q(F1, s), q(F0, s)), Fr(q(F0, s), q(F1, s))) for s in mids if 0 < s < 1)


def test_prop4_equivalences_exact():
    n = 0
    for M in (2, 3, 4):
        L = list(laws(M))
        for mu0 in L:
            for mu1 in L:
                r = rho(mu0, mu1); dsat = math.ceil(r) - 1; assert 0 <= dsat <= M - 1
                first = None
                for D in range(0, 4):
                    b = maxflow(mu0, mu1, D) == 1; c = cond_c(mu0, mu1, D); d = r <= D + 1
                    assert b == c == d, (M, mu0, mu1, D, b, c, d)
                    if b and first is None:
                        first = D
                    n += 1
                assert first == dsat or (first is None and dsat > 3)
    assert n > 5000


def test_delta_endpoints_exact():
    # delta_D = 1 - max matched mass: 0 for identical laws, 1 for laws that no cap-D copying can confuse
    assert 1 - maxflow([Fr(1, 2), Fr(1, 2), Fr(0)], [Fr(1, 2), Fr(1, 2), Fr(0)], 0) == 0
    assert 1 - maxflow([Fr(1), Fr(0), Fr(0), Fr(0)], [Fr(0), Fr(0), Fr(0), Fr(1)], 2) == 1
    assert 1 - maxflow([Fr(1), Fr(0), Fr(0), Fr(0)], [Fr(0), Fr(0), Fr(0), Fr(1)], 3) == 0
