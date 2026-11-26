"""Honest timing: exact rational isotonic solver vs generic convex program (CLARABEL, OSQP).
Same laws, same cap. Warm-up run excluded; 3 repeats, median reported. Seed 2026."""
import time, random, csv, platform, statistics, sys
sys.path.insert(0, ".")
from finite_cap import Game, gen_law_iid, generic_qp
from exact_core import solve_law
import cvxpy, scipy
rows = []
rnd = random.Random(2026)
for K, M in [(1, 20), (2, 4), (3, 3)]:
    if K == 1:
        g = Game(1, M, M)
    else:
        g = Game(K, M, 3 * M + 2)
    Dstar = M - 1 if K == 1 else (2 * M - 3 if K == 2 else 4)
    t0 = time.perf_counter(); _ = [g.thr(n, m) for i, n in enumerate(g.S) for m in g.S[i+1:]]; t_edges = time.perf_counter() - t0
    for tr in range(4):
        law = gen_law_iid(K, M, rnd)
        for D in sorted({1, Dstar}):
            def ex(): return solve_law(law, g.edge_D(D))[0]
            def cl(): return generic_qp(g, law, D, "CLARABEL")
            def oq(): return generic_qp(g, law, D, "OSQP")
            res = {}
            for name, f in [("exact", ex), ("clarabel", cl), ("osqp", oq)]:
                f()
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter(); v = f(); ts.append(time.perf_counter() - t0)
                res[name] = (float(v), statistics.median(ts))
            rows.append(dict(K=K, M=M, law=tr, D=D, atoms=sum(1 for v in law.values() if v > 0),
                             edge_table_build_s=round(t_edges, 3),
                             exact_value=res["exact"][0], exact_s=round(res["exact"][1], 4),
                             clarabel_value=res["clarabel"][0], clarabel_s=round(res["clarabel"][1], 4),
                             osqp_value=res["osqp"][0], osqp_s=round(res["osqp"][1], 4),
                             machine=platform.platform(), python=platform.python_version(),
                             cvxpy=cvxpy.__version__, scipy=scipy.__version__))
            print(rows[-1], flush=True)
with open("out/benchmarks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
