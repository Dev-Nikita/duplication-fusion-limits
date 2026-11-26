"""Exact (rational) core for the whole-report copying game.

Model. K report types, M sources. A firing source emits one nonzero pattern v in {0,1}^K.
State n = (n_v) with sum n_v <= M. Label y in {0,1}. Law p(y,n).
Cap D: every original is delivered 1..D+1 times, so t_v ranges over all integers in [n_v,(D+1)n_v].
Unrestricted: t_v >= n_v, t_v = 0 if n_v = 0. Observation A = sum_v t_v v.
The mechanism sees (y,n). Predictor f(A) in [0,1], Brier loss.

Reduction used here (proof: Supplementary Section S8):
  V(p) = min sum_n p0(n) u_n^2 + sum_m p1(m) (1-l_m)^2  s.t. l_m <= u_n for every edge (n,m),
  edge(n,m) iff R(n) and R(m) intersect.  This is isotonic regression on a bipartite order; it is
  solved exactly by recursive max-closure splitting and certified by an exact KKT flow.
All arithmetic is Fraction / int; floats are used only to *propose* LP witnesses, which are then
verified exactly.
"""
from fractions import Fraction as F
from itertools import product, combinations
from collections import deque
import numpy as np
from scipy.optimize import linprog


def patterns(K):
    return [v for v in product((0, 1), repeat=K) if any(v)]


def states(K, M):
    P = patterns(K)
    out = []

    def rec(i, left, cur):
        if i == len(P):
            out.append(tuple(cur))
            return
        for c in range(left + 1):
            cur.append(c)
            rec(i + 1, left - c, cur)
            cur.pop()
    rec(0, M, [])
    return out  # each state: tuple of counts aligned with patterns(K)


def support(n):
    return tuple(i for i, c in enumerate(n) if c)


# ---------------------------------------------------------------- finite-cap reachable sets
def reach(n, D, P):
    K = len(P[0])
    pts = {tuple([0] * K)}
    for c, v in zip(n, P):
        if c == 0:
            continue
        nxt = set()
        for t in range(c, (D + 1) * c + 1):
            add = tuple(t * x for x in v)
            for a in pts:
                nxt.add(tuple(x + y for x, y in zip(a, add)))
        pts = nxt
    return frozenset(pts)


# ---------------------------------------------------------------- exact cone feasibility
def _nullspace(A):
    """Exact rational nullspace basis of integer matrix A (list of rows)."""
    m = len(A)
    N = len(A[0]) if m else 0
    R = [[F(x) for x in row] for row in A]
    piv = []
    r = 0
    for c in range(N):
        p = next((i for i in range(r, m) if R[i][c] != 0), None)
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        pv = R[r][c]
        R[r] = [x / pv for x in R[r]]
        for i in range(m):
            if i != r and R[i][c] != 0:
                f = R[i][c]
                R[i] = [a - f * b for a, b in zip(R[i], R[r])]
        piv.append(c)
        r += 1
        if r == m:
            break
    free = [c for c in range(N) if c not in piv]
    basis = []
    for fc in free:
        x = [F(0)] * N
        x[fc] = F(1)
        for i, pc in enumerate(piv):
            x[pc] = -R[i][fc]
        basis.append(x)
    return basis


def _project(basis, xf):
    """Exact orthogonal projection of rational vector xf onto span(basis)."""
    k = len(basis)
    if k == 0:
        return None
    G = [[sum(a * b for a, b in zip(basis[i], basis[j])) for j in range(k)] for i in range(k)]
    rhs = [sum(a * b for a, b in zip(basis[i], xf)) for i in range(k)]
    # solve G c = rhs exactly
    M_ = [row[:] + [rhs[i]] for i, row in enumerate(G)]
    for c in range(k):
        p = next(i for i in range(c, k) if M_[i][c] != 0)
        M_[c], M_[p] = M_[p], M_[c]
        pv = M_[c][c]
        M_[c] = [x / pv for x in M_[c]]
        for i in range(k):
            if i != c and M_[i][c] != 0:
                f = M_[i][c]
                M_[i] = [a - f * b for a, b in zip(M_[i], M_[c])]
    coef = [M_[i][k] for i in range(k)]
    N = len(xf)
    return [sum(coef[i] * basis[i][j] for i in range(k)) for j in range(N)]


def relint_meet(Vs, Ws, zbound=4):
    """Decide exactly whether relint cone(Vs) and relint cone(Ws) intersect.
    Vs, Ws: lists of integer vectors (possibly empty; relint cone(empty) = {0}).
    Returns (bool, certificate). Witness: positive rational lam, mu with sum lam v = sum mu w.
    Refutation: integer z with z.v>=0 (v in Vs), z.w<=0 (w in Ws), sum z.v - sum z.w > 0 (Farkas)."""
    K = len((Vs or Ws)[0]) if (Vs or Ws) else 1
    if not Vs and not Ws:
        return True, ("trivial",)
    cols = [list(v) for v in Vs] + [[-x for x in w] for w in Ws]
    N = len(cols)
    A = [[cols[j][i] for j in range(N)] for i in range(K)]
    res = linprog(np.zeros(N), A_eq=np.array(A, float), b_eq=np.zeros(K),
                  bounds=[(1, None)] * N, method="highs")
    if res.status == 0:
        xr = [F(x).limit_denominator(10 ** 6) for x in res.x]
        x = _project(_nullspace(A), xr)
        if x is not None and all(c > 0 for c in x) and all(sum(A[i][j] * x[j] for j in range(N)) == 0 for i in range(K)):
            return True, ("witness", x)
    for z in product(range(-zbound, zbound + 1), repeat=K):
        zv = [sum(a * b for a, b in zip(z, v)) for v in Vs]
        zw = [sum(a * b for a, b in zip(z, w)) for w in Ws]
        if all(c >= 0 for c in zv) and all(c <= 0 for c in zw) and sum(zv) - sum(zw) > 0:
            return False, ("farkas", z)
    raise RuntimeError("no certificate found: increase zbound or LP tolerance")


# ---------------------------------------------------------------- exact isotonic solver
class Flow:
    def __init__(self, n):
        self.n = n
        self.g = [dict() for _ in range(n)]

    def add(self, a, b, c):
        self.g[a][b] = self.g[a].get(b, 0) + c
        self.g[b].setdefault(a, 0)

    def maxflow(self, s, t):
        total = 0
        while True:
            par = {s: None}
            q = deque([s])
            while q and t not in par:
                a = q.popleft()
                for b, c in self.g[a].items():
                    if c > 0 and b not in par:
                        par[b] = a
                        q.append(b)
            if t not in par:
                return total
            path = []
            b = t
            while par[b] is not None:
                path.append((par[b], b))
                b = par[b]
            aug = min(self.g[a][b] for a, b in path)
            for a, b in path:
                self.g[a][b] -= aug
                self.g[b][a] += aug
            total += aug

    def source_side(self, s):
        seen = {s}
        q = deque([s])
        while q:
            a = q.popleft()
            for b, c in self.g[a].items():
                if c > 0 and b not in seen:
                    seen.add(b)
                    q.append(b)
        return seen


INF = None  # marker for infinite capacity


def _closure_split(S, atoms, below, want_flows=False):
    """S: list of atom ids. atoms[i]=(y,w). below[n] = set of label-1 atoms m with l_m <= u_n.
    Returns (theta, L) where L is a max-weight lower set at theta (empty if S is a block),
    plus the saturating flow value check."""
    w1 = sum(atoms[i][1] for i in S if atoms[i][0] == 1)
    wt = sum(atoms[i][1] for i in S)
    theta = w1 / wt
    idx = {a: k + 2 for k, a in enumerate(S)}
    big = wt * 4 + 1  # finite stand-in for infinity (exceeds any cut of finite edges)
    fl = Flow(len(S) + 2)
    pos = 0
    Sset = set(S)
    for i in S:
        y, w = atoms[i]
        g = w * theta if y == 0 else -w * (1 - theta)
        if g > 0:
            fl.add(0, idx[i], g)
            pos += g
        elif g < 0:
            fl.add(idx[i], 1, -g)
        if y == 0:
            for m in below[i]:
                if m in Sset:
                    fl.add(idx[i], idx[m], big)
    f = fl.maxflow(0, 1)
    L = [a for a in S if idx[a] in fl.source_side(0)]
    if want_flows:
        flows = {}
        for i in S:
            if atoms[i][0] == 0:
                for m in below[i]:
                    if m in Sset:
                        used = big - fl.g[idx[i]][idx[m]]
                        if used:
                            flows[(i, m)] = used
        return theta, (L if pos - f > 0 else []), pos == f, flows
    return theta, (L if pos - f > 0 else []), pos == f


def isotonic(atoms, below):
    """Exact solution. Returns dict atom->value (Fraction) and list of blocks."""
    val, blocks = {}, []
    stack = [list(range(len(atoms)))]
    while stack:
        S = stack.pop()
        if not S:
            continue
        theta, L, _ = _closure_split(S, atoms, below)
        if L and len(L) < len(S):
            Ls = set(L)
            stack.append(L)
            stack.append([a for a in S if a not in Ls])
        else:
            for a in S:
                val[a] = theta
            blocks.append(S)
    return val, blocks


def certify(atoms, below, val):
    """Exact KKT certificate: primal feasibility + per-level-set saturating flow."""
    for n, ms in below.items():
        for m in ms:
            if val[m] > val[n]:
                return False
    levels = {}
    for a, v in val.items():
        levels.setdefault(v, []).append(a)
    for v, S in levels.items():
        th, _, sat = _closure_split(S, atoms, below)
        if th != v or not sat:
            return False
    return True


def value(atoms, val):
    return sum(w * (val[i] ** 2 if y == 0 else (1 - val[i]) ** 2) for i, (y, w) in enumerate(atoms))


def build_atoms(law):
    """law: dict (y,state)->Fraction. Returns atoms list and their states."""
    keys = sorted(k for k, w in law.items() if w > 0)
    atoms = [(y, law[(y, n)]) for (y, n) in keys]
    st = [n for (_, n) in keys]
    return atoms, st


def below_from_edges(atoms, st, edge):
    below = {i: set() for i, (y, _) in enumerate(atoms) if y == 0}
    for i in below:
        for j, (y, _) in enumerate(atoms):
            if y == 1 and edge(st[i], st[j]):
                below[i].add(j)
    return below


def solve_law(law, edge):
    atoms, st = build_atoms(law)
    below = below_from_edges(atoms, st, edge)
    val, blocks = isotonic(atoms, below)
    assert certify(atoms, below, val), "KKT certificate failed"
    return value(atoms, val), val, blocks, atoms, st, below


def kkt_multipliers(atoms, below, val):
    """Exact KKT multipliers mu[(n,m)] (flows inside level sets). Raises if a level set has no saturating flow."""
    levels = {}
    for a, v in val.items():
        levels.setdefault(v, []).append(a)
    mu = {}
    for v, S in levels.items():
        th, _, sat, fl = _closure_split(S, atoms, below, want_flows=True)
        if th != v or not sat:
            raise RuntimeError("no certificate")
        mu.update(fl)
    return mu
