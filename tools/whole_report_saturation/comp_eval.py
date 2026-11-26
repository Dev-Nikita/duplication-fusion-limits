"""P3: (a) cold end-to-end timing, (b) amortised many-laws D_sat computation, (c) plug-in D_sat bootstrap.
Usage: python comp_eval.py {cold|amortised|bootstrap} [K M]. Seeds fixed. Machine info recorded."""
import sys, time, random, csv, statistics, platform
from fractions import Fraction as F
from itertools import combinations
from finite_cap import Game, gen_law_iid, generic_qp, dsat_direct
from exact_core import solve_law

def info():
    import cvxpy, scipy
    return dict(machine=platform.platform(), python=platform.python_version(), cvxpy=cvxpy.__version__, scipy=scipy.__version__)

def cold(K, M):
    rnd = random.Random(31); rows = []
    for tr in range(3):
        law = gen_law_iid(K, M, rnd); D = 1
        t0 = time.perf_counter(); g = Game(K, M, 3 * M + 2)
        V = solve_law(law, g.edge_D(D))[0]; te = time.perf_counter() - t0
        t0 = time.perf_counter(); g2 = Game(K, M, 3 * M + 2)
        Vc = generic_qp(g2, law, D, "CLARABEL"); tc = time.perf_counter() - t0
        rows.append(dict(mode="cold_single_solve", K=K, M=M, law=tr, D=D, exact_s=round(te, 3), clarabel_s=round(tc, 3),
                         ratio_clarabel_over_exact=round(tc / te, 2), abs_diff=abs(float(V) - Vc), **info()))
    return rows

def amortised(K, M, L=12):
    rnd = random.Random(32); laws = [gen_law_iid(K, M, rnd) for _ in range(L)]; Dmax = 3 * M + 2
    t0 = time.perf_counter(); g = Game(K, M, Dmax)
    [g.thr(n, m) for n, m in combinations(g.S, 2)]; t_build = time.perf_counter() - t0
    t0 = time.perf_counter(); ds = [dsat_direct(g, law, Dmax)[0] for law in laws]; t_exact = time.perf_counter() - t0
    t0 = time.perf_counter(); g2 = Game(K, M, Dmax); dc = []
    for law in laws:
        Vinf = generic_qp(g2, law, Dmax, "CLARABEL")
        d = next(D for D in range(Dmax + 1) if Vinf - generic_qp(g2, law, D, "CLARABEL") < 1e-7); dc.append(d)
    t_cl = time.perf_counter() - t0
    return [dict(mode="amortised_Dsat", K=K, M=M, laws=L, build_once_s=round(t_build, 2), exact_total_s=round(t_build + t_exact, 2),
                 clarabel_total_s=round(t_cl, 2), ratio_clarabel_over_exact=round(t_cl / (t_build + t_exact), 2),
                 Dsat_agree=(ds == dc), **info())]

def bootstrap(K, M):
    rnd = random.Random(33); rows = []; Dmax = 3 * M + 2; g = Game(K, M, Dmax)
    for li in range(2):
        p = gen_law_iid(K, M, rnd)
        keys = list(p); w = [float(p[k]) for k in keys]
        d_true, Vinf = dsat_direct(g, p, Dmax)
        eps = F(1, 200)
        Vs = [solve_law(p, g.edge_D(D))[0] for D in range(Dmax + 1)]
        de_true = next(D for D in range(Dmax + 1) if Vinf - Vs[D] <= eps)
        for N in (200, 1000, 5000):
            for b in range(12):
                cnt = {}
                for k in rnd.choices(keys, weights=w, k=N):
                    cnt[k] = cnt.get(k, 0) + 1
                ph = {k: F(c, N) for k, c in cnt.items()}
                dh, Vh = dsat_direct(g, ph, Dmax)
                Vsh = [solve_law(ph, g.edge_D(D))[0] for D in range(Dmax + 1)]
                deh = next(D for D in range(Dmax + 1) if Vh - Vsh[D] <= eps)
                rows.append(dict(mode="bootstrap", K=K, M=M, law=li, N=N, rep=b, Dsat_true=d_true, Dsat_hat=dh,
                                 Deps_true=de_true, Deps_hat=deh, eps=float(eps), Vinf_true=float(Vinf), Vinf_hat=float(Vh)))
    return rows

if __name__ == "__main__":
    mode = sys.argv[1]; K, M = int(sys.argv[2]), int(sys.argv[3])
    rows = {"cold": cold, "amortised": amortised, "bootstrap": bootstrap}[mode](K, M)
    fn = f"out/comp_{mode}_K{K}M{M}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows[:3]: print({k: v for k, v in r.items() if k not in ("machine",)})
