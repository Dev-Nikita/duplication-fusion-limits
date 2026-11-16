"""Apply Theorem 2 (exact finite-cap minimax program) to the existing testbed data, with block separation.
ANALYTICAL (no new data, no trained models). Declared 2026-10-26 after all data had been seen; reported as an
analytical application, not as confirmation.

For each dataset, primary task (amf_stall, ausf_stall, net_delay vs healthy) and held-out block b:
  * thresholds: 90th percentile of UE-side latency in the healthy windows of the training blocks (as in R5b-R11);
  * U = number of UEs whose UE-side registration latency exceeds the threshold (report type ue_slow, M = 20);
  * p_train(y, u) = empirical joint law on the training blocks; p_test on the held-out block;
  * for each cap D: the oracle minimax value R*_D(p_train) and the minimax rule x*_D (solver B + primal polishing);
    the out-of-sample guaranteed risk  W_test(x*_D) = exact worst case over Delta_D of the FIXED rule x*_D on p_test;
    the no-copy risk of x*_D on p_test; and the same two quantities for the naive count rule that treats counts as
    unique-source counts (x_a = P_train(y=1 | U = min(a, M)));
  * saturation: R*_D = R_B(P) (checked by the value and, independently, by an LP for conditional independence).
Usage: python3 apply_finite_cap.py <out_dir> <dataset:units.csv:blocks[:cond]> ..."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_solvers as S  # noqa: E402

M = 20; DS = (0, 1, 2, 3, 4, 8, 16); TASKS = ("amf_stall", "ausf_stall", "net_delay")


def polish(x, p, D, sweeps=3, grid=np.linspace(0, 1, 401)):
    """Exact coordinate-wise minimisation of the convex primal (each coordinate is a 1-D convex problem)."""
    I = S.intervals(p.shape[1] - 1, D); A = len(x); members = [[] for _ in range(A)]
    for u, Iu in enumerate(I):
        for a in Iu:
            members[a].append(u)
    x = x.copy()
    for _ in range(sweeps):
        for a in range(A):
            if not members[a]:
                continue
            best, bv = x[a], None
            for v in grid:
                tot = 0.0
                for u in members[a]:
                    others = I[u][I[u] != a]
                    r2 = np.max(x[others] ** 2) if len(others) else 0.0
                    s2 = np.max((1 - x[others]) ** 2) if len(others) else 0.0
                    tot += p[0, u] * max(v * v, r2) + p[1, u] * max((1 - v) ** 2, s2)
                if bv is None or tot < bv - 1e-15:
                    bv, best = tot, v
            x[a] = best
    return x


def lp_saturation(p, D):
    """Independent check: is there an admissible kernel with A independent of Y given presence?"""
    from scipy.optimize import linprog
    I = S.intervals(M, D); A = (D + 1) * M + 1
    m0 = p[0, 1:].sum(); m1 = p[1, 1:].sum()
    if m0 == 0 or m1 == 0:
        return True
    var = []                      # (y, u, a)
    for y in (0, 1):
        for u in range(1, M + 1):
            if p[y, u] > 0:
                for a in I[u]:
                    var.append((y, u, a))
    n = len(var); Aeq = []; beq = []
    for y in (0, 1):              # each (y, u) row of the kernel sums to its mass, normalised within presence
        for u in range(1, M + 1):
            if p[y, u] > 0:
                Aeq.append([1.0 if (v[0] == y and v[1] == u) else 0.0 for v in var]); beq.append(p[y, u] / (m0 if y == 0 else m1))
    for a in range(1, A):         # common conditional law of A given presence
        Aeq.append([(1.0 if v[2] == a else 0.0) * (1 if v[0] == 0 else -1) for v in var]); beq.append(0.0)
    res = linprog(np.zeros(n), A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * n, method="highs")
    return bool(res.status == 0)


def law(df):
    p = np.zeros((2, M + 1))
    for y, u in zip(df.y, df.U):
        p[int(y), min(int(u), M)] += 1
    return p / p.sum()


def windows(units, thr):
    g = units.assign(f=(units.ue_lat > thr).astype(int)).groupby(["episode", "block", "y"]).f.sum().reset_index()
    return g.rename(columns={"f": "U"})


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); rows = []
    for spec in sys.argv[2:]:
        parts = spec.split(":"); name, path, blocks = parts[0], parts[1], [int(b) for b in parts[2].split(",")]
        cond = parts[3] if len(parts) > 3 else None
        u = pd.read_csv(path).replace([np.inf], 1e9)
        u = u[u.block.isin(blocks)]
        if cond:
            u = u[u.load_cond == cond]
        for task in TASKS:
            sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
            for b in blocks:
                tr, te = sub[sub.block != b], sub[sub.block == b]
                if te.y.nunique() < 2:
                    continue
                thr = float(np.percentile(tr[tr.y == 0].ue_lat, 90))
                ptr, pte = law(windows(tr, thr)), law(windows(te, thr))
                eta = np.where(ptr.sum(0) > 0, ptr[1] / np.where(ptr.sum(0) > 0, ptr.sum(0), 1), np.nan)
                eta = pd.Series(eta).ffill().bfill().fillna(0.5).values
                rbp, rbu = S.bayes_risk_presence(ptr), S.bayes_risk_U(ptr)
                for D in DS:
                    lo, up, x, Q = S.solve_dual(ptr, D, iters=4000, tol=1e-7)
                    if up - lo > 1e-4:
                        x = polish(x, ptr, D, sweeps=2, grid=np.linspace(0, 1, 201))
                        up = min(up, S.primal_value(x, ptr, D))
                    naive = eta[np.minimum(np.arange((D + 1) * M + 1), M)]
                    rows.append(dict(dataset=name, task=task, held_block=b, D=D, R_star_lo=lo, R_star_up=up,
                                     RB_U=rbu, RB_P=rbp, saturated_value=bool(lo >= rbp - 1e-4),
                                     saturated_lp=(lp_saturation(ptr, D) if D > 0 else bool(abs(rbu - rbp) < 1e-12)),
                                     W_test_minimax=S.primal_value(x, pte, D), nocopy_test_minimax=S.primal_value(x, pte, 0),
                                     W_test_naive=S.primal_value(naive, pte, D), nocopy_test_naive=S.primal_value(naive, pte, 0),
                                     RB_U_test=S.bayes_risk_U(pte), RB_P_test=S.bayes_risk_presence(pte)))
                print(name, task, b, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "finite_cap_application.csv"), index=False)


if __name__ == "__main__":
    main()
