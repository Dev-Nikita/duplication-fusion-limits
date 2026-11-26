"""Proposition 9: saturation at the presence level for count-only (label-blind) mechanisms, one report type.
Claim: R*(C_A, Delta_D^U) = R_B(P)  iff  the positive and negative parts of mu_0 - mu_1 (laws of U given y, U>0)
can be coupled on pairs u, u' with max(u/u', u'/u) <= D+1;  hence D_sat^U = ceil(rho(nu+, nu-)) - 1 >= D_sat.
Checks (exact rationals for the closed forms):
  random : 300 random laws (M <= 10, all D < M): closed form vs (i) LP feasibility of a common kernel Q with
           mu_0 Q = mu_1 Q (scipy/HiGHS) and (ii) equality of the count-only minimax value (CLARABEL) with R_B(P).
  folds  : D_sat (label-aware, Prop. 4) and D_sat^U on the 108 testbed training laws.
Usage: python3 tools/label_blind_saturation.py random [start end] | folds out.csv"""
import sys, math, random, csv, pickle
from fractions import Fraction as F
import numpy as np


def rho(a, b):
    """quantile ratio sup_s max(q_a/q_b, q_b/q_a) for two distributions on 1..M (dict u->Fraction, sum 1)."""
    us = sorted(set(a) | set(b))
    ca, cb = [], []
    sa = sb = F(0)
    for u in us:
        sa += a.get(u, F(0)); sb += b.get(u, F(0)); ca.append((sa, u)); cb.append((sb, u))
    bps = sorted(set(c for c, _ in ca) | set(c for c, _ in cb))
    q = lambda C, s: next(u for c, u in C if c >= s)
    r, prev = F(1), F(0)
    for bp in bps:
        if bp == prev:
            continue
        s = (prev + bp) / 2; x, y = q(ca, s), q(cb, s); r = max(r, F(x, y), F(y, x)); prev = bp
    return r


def laws(n):
    """n: 2 x (M+1) counts. Returns mu_0, mu_1 on u >= 1 (dicts of Fractions) or None if a class has no mass there."""
    mu = []
    for y in (0, 1):
        tot = sum(int(v) for v in n[y][1:])
        if tot == 0:
            return None
        mu.append({u: F(int(n[y][u]), tot) for u in range(1, len(n[y])) if n[y][u]})
    return mu


def dsat_aware(mu0, mu1):
    return math.ceil(rho(mu0, mu1)) - 1


def dsat_blind(mu0, mu1):
    us = set(mu0) | set(mu1); nu = {u: mu0.get(u, F(0)) - mu1.get(u, F(0)) for u in us}
    t = sum(v for v in nu.values() if v > 0)
    if t == 0:
        return 0
    pos = {u: v / t for u, v in nu.items() if v > 0}; neg = {u: -v / t for u, v in nu.items() if v < 0}
    return math.ceil(rho(pos, neg)) - 1


def lp_common_kernel(mu0, mu1, D, M):
    from scipy.optimize import linprog
    us = sorted(set(mu0) | set(mu1)); var = [(u, a) for u in us for a in range(u, (D + 1) * u + 1)]
    idx = {v: i for i, v in enumerate(var)}; A = (D + 1) * M + 1
    Aeq, beq = [], []
    for u in us:
        row = np.zeros(len(var)); row[[idx[(u, a)] for a in range(u, (D + 1) * u + 1)]] = 1; Aeq.append(row); beq.append(1)
    for a in range(1, A):
        row = np.zeros(len(var))
        for u in us:
            if (u, a) in idx:
                row[idx[(u, a)]] = float(mu0.get(u, 0) - mu1.get(u, 0))
        if row.any():
            Aeq.append(row); beq.append(0)
    r = linprog(np.zeros(len(var)), A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * len(var), method="highs")
    return r.status == 0


def countonly_value(p, D):
    import cvxpy as cp
    M = p.shape[1] - 1; n = (D + 1) * M + 1; x = cp.Variable(n); terms = []
    for u in range(M + 1):
        I = [0] if u == 0 else list(range(u, (D + 1) * u + 1))
        terms.append(cp.max(cp.hstack([p[0, u] * cp.square(x[a]) + p[1, u] * cp.square(1 - x[a]) for a in I])))
    pr = cp.Problem(cp.Minimize(cp.sum(cp.hstack(terms))), [x >= 0, x <= 1]); pr.solve(solver="CLARABEL")
    return pr.value


def rb_presence(p):
    v = 0.0
    for cols in ([0], list(range(1, p.shape[1]))):
        w0, w1 = p[0, cols].sum(), p[1, cols].sum()
        v += w0 * w1 / (w0 + w1) if w0 + w1 > 0 else 0
    return v


def mode_random(start=0, end=300):
    rnd = random.Random(20261125); checks = bad_lp = bad_cvx = 0; strict = 0
    for k in range(300):
        M = rnd.randint(2, 10); n = np.zeros((2, M + 1), int)
        for y in (0, 1):
            for u in range(M + 1):
                if rnd.random() < 0.55:
                    n[y, u] = rnd.randint(1, 9)
            if n[y, 1:].sum() == 0:
                n[y, rnd.randint(1, M)] = 1
        if not start <= k < end:
            continue
        mu0, mu1 = laws(n); da, db = dsat_aware(mu0, mu1), dsat_blind(mu0, mu1)
        assert db >= da
        strict += db > da
        p = n / n.sum(); rbp = rb_presence(p)
        for D in range(0, M):
            closed = D >= db; checks += 1
            if lp_common_kernel(mu0, mu1, D, M) != closed:
                bad_lp += 1
            if (abs(countonly_value(p, D) - rbp) < 1e-6) != closed:
                bad_cvx += 1
    print(f"Prop. 9: random laws {start}-{end - 1} of 300, {checks} law-cap cases: closed form vs common-kernel LP {bad_lp} mismatches, "
          f"vs count-only minimax value (CLARABEL, tol 1e-6) {bad_cvx} mismatches; D_sat^U > D_sat in {strict} laws")


def mode_folds(out):
    rows = []
    for f in pickle.load(open("results/finite_cap/fold_laws.pkl", "rb")):
        mu = laws(f["n_train"])
        if mu is None:
            continue
        rows.append(dict(dataset=f["dataset"], task=f["task"], held_block=f["held_block"],
                         D_sat_aware=dsat_aware(*mu), D_sat_countonly=dsat_blind(*mu)))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    d = np.array([r["D_sat_countonly"] - r["D_sat_aware"] for r in rows])
    print(f"folds {len(rows)}: D_sat^U - D_sat: equal in {(d == 0).sum()}, larger in {(d > 0).sum()} (median gap among larger {np.median(d[d > 0]) if (d > 0).any() else 0})")


if __name__ == "__main__":
    if sys.argv[1] == "random":
        mode_random(*(int(a) for a in sys.argv[2:4])) if len(sys.argv) > 2 else mode_random()
    else:
        mode_folds(sys.argv[2])
