#!/usr/bin/env python3
"""Independent numerical oracle for the finite-copy minimax theorem.

This file intentionally imports nothing from src/paef.  It solves (i) the
epigraph primal and (ii) the least-favourable-channel problem independently.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from scipy.optimize import minimize


def intervals(M: int, D: int) -> list[list[int]]:
    return [[0] if u == 0 else list(range(u, (D + 1) * u + 1)) for u in range(M + 1)]


def solve_primal(p: np.ndarray, D: int) -> dict:
    M = p.shape[1] - 1
    A = (D + 1) * M + 1
    ids = intervals(M, D)
    # z = [x_0..x_{A-1}, r_0..r_M, s_0..s_M]
    n = A + 2 * (M + 1)
    def obj(z):
        r, s = z[A:A + M + 1], z[A + M + 1:]
        return float(p[0] @ (r * r) + p[1] @ ((1 - s) ** 2))
    cons = []
    for u, aa in enumerate(ids):
        for a in aa:
            cons.append({"type": "ineq", "fun": lambda z, u=u, a=a: z[A + u] - z[a]})
            cons.append({"type": "ineq", "fun": lambda z, u=u, a=a: z[a] - z[A + M + 1 + u]})
    z0 = np.r_[np.full(A, .5), np.full(M + 1, .5), np.full(M + 1, .5)]
    out = minimize(obj, z0, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=cons, options={"ftol": 1e-12, "maxiter": 5000})
    if not out.success:
        raise RuntimeError(out.message)
    return {"value": obj(out.x), "x": out.x[:A].tolist(), "solver": out.message}


def solve_channel(p: np.ndarray, D: int) -> dict:
    M = p.shape[1] - 1
    A = (D + 1) * M + 1
    ids = intervals(M, D)
    cells = [(y, u, a) for y in (0, 1) for u in range(M + 1) for a in ids[u]]
    by_yu = {(y, u): [i for i, c in enumerate(cells) if c[:2] == (y, u)]
             for y in (0, 1) for u in range(M + 1)}
    def masses(q):
        w = np.zeros((2, A))
        for qi, (y, u, a) in zip(q, cells):
            w[y, a] += p[y, u] * qi
        return w
    def bayes(q):
        w = masses(q); den = w.sum(axis=0)
        return float(np.divide(w[0] * w[1], den, out=np.zeros(A), where=den > 0).sum())
    constraints = [{"type": "eq", "fun": lambda q, ii=ii: q[ii].sum() - 1}
                   for ii in by_yu.values()]
    q0 = np.zeros(len(cells))
    for ii in by_yu.values():
        q0[ii] = 1 / len(ii)
    out = minimize(lambda q: -bayes(q), q0, method="SLSQP", bounds=[(0, 1)] * len(q0),
                   constraints=constraints, options={"ftol": 1e-12, "maxiter": 10000})
    if not out.success:
        raise RuntimeError(out.message)
    w = masses(out.x); den = w.sum(axis=0)
    x = np.divide(w[1], den, out=np.full(A, .5), where=den > 0)
    q_rows = [{"y": y, "u": u, "a": a, "prob": float(out.x[i])}
              for i, (y, u, a) in enumerate(cells) if out.x[i] > 1e-8]
    return {"value": bayes(out.x), "x": x.tolist(), "Q_nonzero": q_rows,
            "solver": out.message}


def strict_example() -> np.ndarray:
    # Rows are y=0,1; entries are joint masses P(y,u), u=0..3.
    return np.array([[0, .25, 0, .25], [0, 0, .5, 0]], dtype=float)


def bayes_u(p: np.ndarray) -> float:
    den = p.sum(axis=0)
    return float(np.divide(p[0] * p[1], den, out=np.zeros_like(den), where=den > 0).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    ap.add_argument("--random-cases", type=int, default=20)
    args = ap.parse_args()
    p = strict_example()
    primal, channel = solve_primal(p, 1), solve_channel(p, 1)
    gap = abs(primal["value"] - channel["value"])
    rng = np.random.default_rng(20261024)
    random_checks = []
    for M in (1, 2, 3, 4):
        for D in (0, 1, 2):
            for _ in range(args.random_cases):
                p0 = rng.dirichlet(np.ones(2 * (M + 1))).reshape(2, M + 1)
                rp, rq = solve_primal(p0, D), solve_channel(p0, D)
                item = {"M": M, "D": D, "primal": rp["value"], "channel": rq["value"],
                        "gap": abs(rp["value"] - rq["value"])}
                if D == 0:
                    item["bayes_U"] = bayes_u(p0)
                    item["D0_gap"] = abs(rp["value"] - item["bayes_U"])
                random_checks.append(item)
    max_gap = max(x["gap"] for x in random_checks)
    max_d0_gap = max(x.get("D0_gap", 0) for x in random_checks)
    report = {"case": "M=3,D=1 strict example", "primal": primal,
              "least_favourable_channel": channel, "absolute_gap": gap,
              "random_regression": {"cases": len(random_checks), "max_primal_dual_gap": max_gap,
                                    "max_D0_vs_bayes_U_gap": max_d0_gap},
              "checks": {"values_agree": gap < 1e-8,
                         "exact_value_is_one_quarter": abs(primal["value"] - .25) < 1e-8,
                         "random_values_agree": max_gap < 1e-7,
                         "D0_equals_bayes_U": max_d0_gap < 1e-7,
                         "interval_partition_upper_value": 1 / 6}}
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    if not all(v for k, v in report["checks"].items() if isinstance(v, bool)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
