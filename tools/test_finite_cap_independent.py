"""Verification tests for Theorem 2 with the independent solvers. Usage: python3 tools/test_finite_cap_independent.py
Writes a PASS/FAIL table to stdout."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_solvers as S  # noqa: E402

rows = []


def check(name, cond, detail):
    rows.append((name, "PASS" if cond else "FAIL", detail)); print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}", flush=True)


# 1. Counterexample of the manuscript: M = 3, D = 1, exact value 1/4 = R_B(P), R_B(U) = 0
p = np.zeros((2, 4)); p[0, 1] = p[0, 3] = 0.25; p[1, 2] = 0.5
lo, up, x, Q = S.solve_dual(p, 1, iters=50000, tol=1e-10)
vA, xA = S.solve_primal(p, 1)
check("M3D1 dual (B) brackets 1/4", lo <= 0.25 + 1e-9 and up >= 0.25 - 1e-9 and up - lo < 1e-4, f"lower {lo:.6f} upper {up:.6f}")
check("M3D1 primal (A) = 1/4", abs(vA - 0.25) < 1e-5, f"A {vA:.6f}")
check("M3D1 R_B(U)=0, R_B(P)=1/4", abs(S.bayes_risk_U(p)) < 1e-12 and abs(S.bayes_risk_presence(p) - 0.25) < 1e-12,
      f"R_B(U) {S.bayes_risk_U(p):.3f}, R_B(P) {S.bayes_risk_presence(p):.3f}")
# the hand-made kernel of the manuscript reaches 1/4
Q0 = [np.array([1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 0, 0, 0])]
Q1 = [np.array([1.0]), np.array([1.0, 0.0]), np.array([0.5, 0.5, 0.0]), np.array([1.0, 0, 0, 0])]
check("M3D1 manuscript kernel gives 1/4", abs(S.dual_value(Q0, Q1, p, 1)[0] - 0.25) < 1e-12, "hand kernel")

# 2. Random laws: D = 0 equals R_B(U); D >= M-1 equals R_B(P); A and B agree; monotone in D
rng = np.random.default_rng(20261026)
for trial in range(12):
    M = int(rng.integers(2, 7)); p = rng.dirichlet(np.ones(2 * (M + 1))).reshape(2, M + 1)
    vals = []
    for D in range(0, M + 1):
        lo, up, x, Q = S.solve_dual(p, D, iters=30000, tol=1e-8)
        vA, _ = S.solve_primal(p, D)
        vals.append((lo, up, vA))
        check(f"rand{trial} M{M} D{D} A within B-bracket", lo - 1e-6 <= vA <= up + 1e-6 and up - lo < 2e-3,
              f"B [{lo:.5f}, {up:.5f}], A {vA:.5f}")
    check(f"rand{trial} D=0 equals R_B(U)", abs(vals[0][2] - S.bayes_risk_U(p)) < 1e-5, f"{vals[0][2]:.6f} vs {S.bayes_risk_U(p):.6f}")
    check(f"rand{trial} D=M-1 equals R_B(P)", abs(vals[M - 1][2] - S.bayes_risk_presence(p)) < 1e-5,
          f"{vals[M - 1][2]:.6f} vs {S.bayes_risk_presence(p):.6f}")
    mono = all(vals[i + 1][2] >= vals[i][2] - 1e-6 for i in range(len(vals) - 1))
    check(f"rand{trial} non-decreasing in D", mono, " ".join(f"{v[2]:.4f}" for v in vals))

n_fail = sum(r[1] == "FAIL" for r in rows)
print(f"\n{len(rows)} checks, {n_fail} FAIL")
