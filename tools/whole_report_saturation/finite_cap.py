"""G1/G3 study: exact finite-cap saturation for whole-report copying.

Usage: python finite_cap.py MODE
  edges   : state-level edge thresholds d(n,m), D*(K,M), unrestricted-edge consistency (K=1..3)
  k2form  : closed-form K=2 edge rule vs exact enumeration
  k3axis  : real-zonotope (LP) intersection vs exact lattice intersection, K=3
  k1prop4 : K=1 saturation cap vs Proposition 4 closed form
  laws    : random + generative laws, D_sat direct vs criterion vs common-point cap
  numeric : exact values vs CLARABEL and OSQP on the generic program
Seeds are fixed. All decisive comparisons are exact (Fraction/int).
"""
import sys, os, time, csv, random, json, math
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from exact_core import (patterns, states, support, reach, relint_meet, solve_law,
                        build_atoms, below_from_edges, isotonic, certify, value, _closure_split)

OUT = os.path.join(os.getcwd(), "out")  # run from results/whole_report


class Game:
    def __init__(self, K, M, Dmax):
        self.K, self.M, self.Dmax = K, M, Dmax
        self.P = patterns(K)
        self.S = states(K, M)
        self._reach = {}
        self._sup = {}
        self._thr = {}

    def R(self, n, D):
        k = (n, D)
        if k not in self._reach:
            self._reach[k] = reach(n, D, self.P)
        return self._reach[k]

    def sup_edge(self, V, W):
        k = (V, W) if V <= W else (W, V)
        if k not in self._sup:
            ok, cert = relint_meet([self.P[i] for i in V], [self.P[i] for i in W])
            self._sup[k] = (ok, cert)
        return self._sup[k][0]

    def inf_edge(self, n, m):
        return self.sup_edge(support(n), support(m))

    def thr(self, n, m):
        """least D with R_D(n) & R_D(m) nonempty; None if none up to Dmax."""
        k = (n, m) if n <= m else (m, n)
        if k not in self._thr:
            t = None
            for D in range(self.Dmax + 1):
                if not self.R(n, D).isdisjoint(self.R(m, D)):
                    t = D
                    break
            self._thr[k] = t
        return self._thr[k]

    def edge_D(self, D):
        return lambda n, m: (self.thr(n, m) is not None and self.thr(n, m) <= D)


# ------------------------------------------------------------------ mode: edges
def mode_edges():
    rows = []
    for K, Ms in [(1, range(1, 9)), (2, range(1, 6)), (3, range(1, 4))]:
        for M in Ms:
            t0 = time.time()
            Dmax = 3 * M + 2
            g = Game(K, M, Dmax)
            dstar, bad, nedge, worst = 0, 0, 0, None
            for n, m in combinations(g.S, 2):
                e_inf = g.inf_edge(n, m)
                t = g.thr(n, m)
                if e_inf != (t is not None):
                    bad += 1
                if t is not None:
                    nedge += 1
                    if t > dstar:
                        dstar, worst = t, (n, m)
            rows.append(dict(K=K, M=M, states=len(g.S), inf_edges=nedge, mismatches_inf_vs_finite=bad,
                             Dstar_uniform=dstar, worst_pair=json.dumps(worst), Dmax_searched=Dmax,
                             seconds=round(time.time() - t0, 2)))
            print(rows[-1], flush=True)
    with open(f"{OUT}/edges_Dstar.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ mode: K=2 closed form
def ratio(a, b):
    if a == 0 and b == 0:
        return F(1)
    if a == 0 or b == 0:
        return None  # infinite
    return max(F(a, b), F(b, a))


def k2_rho(n, m):
    # patterns(2) order: (0,1),(1,0),(1,1)
    n01, n10, n11 = n
    m01, m10, m11 = m
    rs = [ratio(n10 + n11, m10 + m11), ratio(n01 + n11, m01 + m11), ratio(n10 + m01, m10 + n01)]
    if any(r is None for r in rs):
        return None
    return max(rs)


def mode_k2form():
    assert patterns(2) == [(0, 1), (1, 0), (1, 1)]
    checks = bad = 0
    for M in range(1, 6):
        g = Game(2, M, 3 * M + 2)
        for n, m in combinations(g.S, 2):
            rho = k2_rho(n, m)
            pred = None if rho is None else max(0, math.ceil(rho) - 1)
            checks += 1
            if pred != g.thr(n, m):
                bad += 1
                print("MISMATCH", n, m, pred, g.thr(n, m))
    print(f"K=2 closed form: {checks} state pairs (M<=5), {bad} mismatches")
    with open(f"{OUT}/k2_closed_form.txt", "w") as f:
        f.write(f"K=2 closed-form edge threshold vs exact enumeration: {checks} pairs, M<=5, {bad} mismatches\n")


# ------------------------------------------------------------------ mode: K=3 real vs lattice
def lp_meet(n, m, D, P):
    from scipy.optimize import linprog
    K = len(P[0])
    Vs = [i for i, c in enumerate(n) if c]
    Ws = [i for i, c in enumerate(m) if c]
    N = len(Vs) + len(Ws)
    A = np.zeros((K, N))
    bnds = []
    for j, i in enumerate(Vs):
        A[:, j] = P[i]; bnds.append((n[i], (D + 1) * n[i]))
    for j, i in enumerate(Ws):
        A[:, len(Vs) + j] = -np.array(P[i]); bnds.append((m[i], (D + 1) * m[i]))
    if N == 0:
        return True
    r = linprog(np.zeros(N), A_eq=A, b_eq=np.zeros(K), bounds=bnds, method="highs")
    return r.status == 0


def mode_k3axis():
    rows = []
    for M in (2, 3):
        g = Game(3, M, 3 * M + 2)
        gaps = []
        for D in range(0, 3 * M + 1):
            for n, m in combinations(g.S, 2):
                real = lp_meet(n, m, D, g.P)
                lat = not g.R(n, D).isdisjoint(g.R(m, D))
                if lat and not real:
                    raise RuntimeError("lattice point without real point: impossible")
                if real and not lat:
                    gaps.append((D, n, m))
        rows.append((M, len(gaps), gaps[:5]))
        print(f"K=3 M={M}: pairs with real (LP) but no lattice intersection: {len(gaps)}; examples {gaps[:3]}")
    with open(f"{OUT}/k3_lattice_gaps.txt", "w") as f:
        for M, k, ex in rows:
            f.write(f"K=3 M={M}: {k} (D,n,m) triples where the real zonotopes meet but no common lattice point; examples {ex}\n")
        f.write("Pattern order: " + str(patterns(3)) + "\n")


# ------------------------------------------------------------------ mode: K=1 vs Prop 4
def prop4_rho(law, M):
    p = {y: {u: law.get((y, (u,)), F(0)) for u in range(1, M + 1)} for y in (0, 1)}
    tot = {y: sum(p[y].values()) for y in (0, 1)}
    if tot[0] == 0 or tot[1] == 0:
        return None
    # quantile ratio sup over s
    cum = {y: [] for y in (0, 1)}
    for y in (0, 1):
        c = F(0)
        for u in range(1, M + 1):
            c += p[y][u] / tot[y]
            cum[y].append((c, u))
    bps = sorted(set(c for y in (0, 1) for c, _ in cum[y]))
    def q(y, s):
        return next(u for c, u in cum[y] if c >= s)
    rho = F(1)
    prev = F(0)
    for b in bps:
        s = (prev + b) / 2  # interior of each constancy interval
        r = max(F(q(1, s), q(0, s)), F(q(0, s), q(1, s)))
        rho = max(rho, r)
        prev = b
    return rho


def dsat_direct(g, law, Dmax):
    Vinf = solve_law(law, g.inf_edge)[0]
    for D in range(Dmax + 1):
        if solve_law(law, g.edge_D(D))[0] == Vinf:
            return D, Vinf
    return None, Vinf


def mode_k1prop4():
    rnd = random.Random(20261117)
    checks = bad = 0
    for trial in range(300):
        M = rnd.randint(2, 10)
        g = Game(1, M, M)
        law = {}
        for y in (0, 1):
            for u in range(0, M + 1):
                if rnd.random() < 0.6:
                    law[(y, (u,))] = F(rnd.randint(1, 9))
        for y in (0, 1):  # both classes need U>0 mass
            if not any(k[0] == y and k[1][0] > 0 for k in law):
                law[(y, (rnd.randint(1, M),))] = F(1)
        Z = sum(law.values()); law = {k: v / Z for k, v in law.items()}
        rho = prop4_rho(law, M)
        pred = math.ceil(rho) - 1
        d, _ = dsat_direct(g, law, M)
        checks += 1
        if d != pred:
            bad += 1
            print("MISMATCH", law, rho, d)
    msg = f"K=1: exact D_sat (isotonic, min D with V_D=V_inf) vs Prop.4 ceil(rho)-1: {checks} random laws (M<=10), {bad} mismatches"
    print(msg)
    open(f"{OUT}/k1_prop4.txt", "w").write(msg + "\n")


# ------------------------------------------------------------------ mode: laws
def common_point_cap(g, Sstates, Dmax):
    for D in range(Dmax + 1):
        inter = None
        for n in Sstates:
            r = g.R(n, D)
            inter = r if inter is None else inter & r
            if not inter:
                break
        if inter:
            return D
    return None


def criterion_dsat(g, law, Dmax):
    """Transport criterion: least D such that inside every unrestricted level set the KKT flow
    exists on E_D edges."""
    atoms, st = build_atoms(law)
    below_inf = below_from_edges(atoms, st, g.inf_edge)
    val, _ = isotonic(atoms, below_inf)
    levels = {}
    for a, v in val.items():
        levels.setdefault(v, []).append(a)
    need = 0
    blocks_info = []
    for v, S in levels.items():
        if v in (0, 1) or len(S) == 1:
            continue
        found = None
        for D in range(Dmax + 1):
            below = below_from_edges(atoms, st, g.edge_D(D))
            _, _, sat = _closure_split(S, atoms, below)
            if sat:
                found = D
                break
        blocks_info.append((v, [st[a] for a in S], found))
        if found is None:
            return None, blocks_info, val, atoms, st
        need = max(need, found)
    return need, blocks_info, val, atoms, st


def gen_law_iid(K, M, rnd):
    """Generative law: y ~ Bern(1/2); each of M sources independently silent w.p. s_y, else emits
    pattern v w.p. r_y(v). Exact multinomial probabilities."""
    P = patterns(K)
    law = {}
    for y in (0, 1):
        s = F(rnd.randint(1, 8), 10)
        wts = [rnd.randint(0, 5) for _ in P]
        if sum(wts) == 0:
            wts[rnd.randrange(len(P))] = 1
        r = [F(w, sum(wts)) for w in wts]
        for n in states(K, M):
            k = M - sum(n)
            coef = math.factorial(M) // math.factorial(k)
            pr = F(coef) * s ** k
            for c, rv in zip(n, r):
                pr = pr * ((1 - s) * rv) ** c / math.factorial(c)
            if pr > 0:
                law[(y, n)] = pr / 2
    assert sum(law.values()) == 1, sum(law.values())
    return law


def gen_law_sparse(K, M, rnd, size):
    S = states(K, M)
    law = {}
    for _ in range(size):
        law[(rnd.randint(0, 1), rnd.choice(S))] = F(rnd.randint(1, 9))
    Z = sum(law.values())
    return {k: v / Z for k, v in law.items()}


def mode_laws():
    rnd = random.Random(1117)
    rows = []
    configs = [(2, 3, "iid", 40), (2, 4, "iid", 30), (2, 3, "sparse", 60), (2, 4, "sparse", 40),
               (3, 2, "iid", 30), (3, 2, "sparse", 60), (3, 3, "sparse", 40), (3, 3, "iid", 15)]
    for K, M, kind, ntr in configs:
        Dmax = 3 * M + 2
        g = Game(K, M, Dmax)
        dstar = max((g.thr(n, m) for n, m in combinations(g.S, 2) if g.thr(n, m) is not None), default=0)
        for tr in range(ntr):
            t0 = time.time()
            law = gen_law_iid(K, M, rnd) if kind == "iid" else gen_law_sparse(K, M, rnd, rnd.randint(4, 14))
            dd, Vinf = dsat_direct(g, law, Dmax)
            dc, binfo, val, atoms, st = criterion_dsat(g, law, Dmax)
            cps = [common_point_cap(g, S, Dmax) for _, S, _ in binfo]
            dcp = None if any(c is None for c in cps) else max(cps, default=0)
            V0 = solve_law(law, g.edge_D(0))[0]
            rows.append(dict(K=K, M=M, kind=kind, trial=tr, atoms=len(atoms), blocks=len(binfo),
                             V0=float(V0), Vinf=float(Vinf), Vinf_exact=str(Vinf),
                             Dsat_direct=dd, Dsat_criterion=dc, agree=(dd == dc),
                             Dcommonpoint=("none<=%d" % Dmax if dcp is None else dcp),
                             Dstar_uniform=dstar, seconds=round(time.time() - t0, 2)))
            print(rows[-1], flush=True)
    with open(f"{OUT}/laws_dsat.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ mode: numeric cross-check
def generic_qp(g, law, D, solver):
    import cvxpy as cp
    pts = sorted(set().union(*[g.R(n, D) for (_, n) in law]))
    ix = {a: i for i, a in enumerate(pts)}
    x = cp.Variable(len(pts))
    keys = list(law)
    r = cp.Variable(len(keys)); s = cp.Variable(len(keys))
    cons = [x >= 0, x <= 1]
    obj = 0
    for k, (y, n) in enumerate(keys):
        idx = [ix[a] for a in g.R(n, D)]
        if y == 0:
            cons.append(r[k] >= x[idx]); obj += float(law[(y, n)]) * cp.square(r[k])
        else:
            cons.append(s[k] <= x[idx]); obj += float(law[(y, n)]) * cp.square(1 - s[k])
    pr = cp.Problem(cp.Minimize(obj), cons)
    pr.solve(solver=solver)
    return pr.value


def mode_numeric():
    rnd = random.Random(99)
    rows = []
    for K, M in [(1, 6), (2, 3), (3, 2)]:
        g = Game(K, M, 3 * M + 2)
        for tr in range(8):
            law = gen_law_sparse(K, M, rnd, rnd.randint(5, 12))
            for D in (0, 1, 2, M):
                ex = solve_law(law, g.edge_D(D))[0]
                a = generic_qp(g, law, D, "CLARABEL")
                b = generic_qp(g, law, D, "OSQP")
                rows.append(dict(K=K, M=M, trial=tr, D=D, exact=str(ex), exact_f=float(ex),
                                 clarabel=a, osqp=b, err_clarabel=abs(a - float(ex)), err_osqp=abs(b - float(ex))))
    with open(f"{OUT}/numeric_crosscheck.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("max |exact-CLARABEL| =", max(r["err_clarabel"] for r in rows),
          " max |exact-OSQP| =", max(r["err_osqp"] for r in rows), " n =", len(rows))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"edges": mode_edges, "k2form": mode_k2form, "k3axis": mode_k3axis, "k1prop4": mode_k1prop4,
     "laws": mode_laws, "numeric": mode_numeric}[sys.argv[1]]()
