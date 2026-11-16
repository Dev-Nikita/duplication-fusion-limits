"""Closed-form saturation criterion for the finite-cap game, checked against the LP of Proposition 3.

Claim: with mu_y = law of U given (y, U>0) and F_y its CDF on {1..M}, R*(C_A, Delta_D) = R_B(P) iff
    F_0(t) <= F_1((D+1) t)  and  F_1(t) <= F_0((D+1) t)   for every t >= 1,
equivalently  sup_{p in (0,1)} max(q_1(p)/q_0(p), q_0(p)/q_1(p)) <= D+1  (left-continuous quantiles), so
    D_sat = ceil(rho) - 1,   rho = sup_p max(q_1(p)/q_0(p), q_0(p)/q_1(p)).
Usage: python3 tools/saturation_closed_form.py [--quick]   (prints PASS/FAIL counts; --quick: 30 laws)
       python3 tools/saturation_closed_form.py --folds results/finite_cap/fold_laws.pkl results/finite_cap/saturation_closed_form.csv"""
import os, sys, math
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_solvers as S  # noqa: E402


def cond_laws(p):
    M = p.shape[1] - 1; m = p[:, 1:].sum(1)
    return p[0, 1:] / m[0], p[1, 1:] / m[1], M


def rho(p):
    """sup over p in (0,1) of the quantile ratio; quantiles are step functions, so check all breakpoints."""
    mu0, mu1, M = cond_laws(p); F0, F1 = np.cumsum(mu0), np.cumsum(mu1)
    def q(F, s):                                   # left-continuous quantile: min{u : F(u) >= s}
        return int(np.searchsorted(F, s - 1e-12) + 1)
    br = np.unique(np.r_[F0, F1]); br = br[(br > 1e-12) & (br < 1 - 1e-12)]
    pts = np.r_[0.5 * (np.r_[0, br] + np.r_[br, 1])]  # one point in every interval between breakpoints
    r = 1.0
    for s in pts:
        a, b = q(F0, s), q(F1, s); r = max(r, a / b, b / a)
    return r


def closed_form_saturated(p, D):
    mu0, mu1, M = cond_laws(p); F0, F1 = np.cumsum(mu0), np.cumsum(mu1)
    for t in range(1, M + 1):
        c = min((D + 1) * t, M)
        if F0[t - 1] > F1[c - 1] + 1e-12 or F1[t - 1] > F0[c - 1] + 1e-12:
            return False
    return True


def lp_saturated(p, D):
    mu0, mu1, M = cond_laws(p); I = S.intervals(M, D); A = (D + 1) * M + 1
    var = [(y, u, a) for y, mu in ((0, mu0), (1, mu1)) for u in range(1, M + 1) if mu[u - 1] > 0 for a in I[u]]
    Aeq, beq = [], []
    for y, mu in ((0, mu0), (1, mu1)):
        for u in range(1, M + 1):
            if mu[u - 1] > 0:
                Aeq.append([1.0 if (v[0] == y and v[1] == u) else 0.0 for v in var]); beq.append(mu[u - 1])
    for a in range(1, A):
        Aeq.append([(1.0 if v[2] == a else 0.0) * (1 if v[0] == 0 else -1) for v in var]); beq.append(0.0)
    res = linprog(np.zeros(len(var)), A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * len(var), method="highs")
    return res.status == 0


def folds_mode(pkl, out):
    """D_sat for every fold law (results/finite_cap/saturation_closed_form.csv)."""
    import pickle, pandas as pd
    rows = []
    for f in pickle.load(open(pkl, "rb")):
        p = f["n_train"] / f["n_train"].sum(); r = rho(p)
        rows.append(dict(dataset=f["dataset"], task=f["task"], held_block=f["held_block"], rho=r, D_sat_closed=math.ceil(r - 1e-12) - 1))
    pd.DataFrame(rows).to_csv(out, index=False); print(len(rows), "folds ->", out)


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--folds":
    folds_mode(sys.argv[2], sys.argv[3]); sys.exit(0)

if __name__ == "__main__":
    TRIALS = 30 if "--quick" in sys.argv else 300
    rng = np.random.default_rng(20261027); n = fail = 0
    for trial in range(TRIALS):
        M = int(rng.integers(2, 13))
        k = rng.integers(1, M + 1, size=2)       # sparse supports make non-saturation frequent
        p = np.zeros((2, M + 1))
        for y in (0, 1):
            sup = rng.choice(np.arange(1, M + 1), size=k[y], replace=False)
            p[y, sup] = rng.dirichlet(np.ones(k[y])); p[y, 0] = rng.random()
        p /= p.sum()
        r = rho(p); dsat = math.ceil(r - 1e-12) - 1
        for D in range(0, M):
            a, b = closed_form_saturated(p, D), lp_saturated(p, D); n += 1; fail += (a != b)
            c = (D >= dsat); n += 1; fail += (c != b)
    # support-overlap counterexample: M=3, D=1
    p = np.zeros((2, 4)); p[0, 1] = 0.5; p[1, 1] = 0.25; p[1, 3] = 0.25
    lo, up, x, Q = S.solve_dual(p, 1, iters=20000, tol=1e-10)
    print(f"counterexample: LP {lp_saturated(p, 1)}, closed form {closed_form_saturated(p, 1)}, rho {rho(p):.2f}, "
          f"R*_1 in [{lo:.5f},{up:.5f}], R_B(P) {S.bayes_risk_presence(p):.5f}, R_B(U) {S.bayes_risk_U(p):.5f}")
    print(f"{n} checks, {fail} FAIL")
