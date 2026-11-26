"""Independent exact verifier of the KKT certificates written by cert_audit.py.
Does not import the solver. Recomputes reachable sets by its own enumeration (unrestricted case: cap 3M+2,
which equals the unrestricted adjacency by out/edges_Dstar.csv), and checks exactly:
 primal feasibility l_m <= u_n on every adjacent pair; mu >= 0 and supported on adjacent pairs;
 stationarity w_n u_n = sum_m mu_nm and w_m (1 - l_m) = sum_n mu_nm; complementary slackness;
 value = sum w * loss. Usage: python kkt_verify.py out/certs/cfg*.jsonl"""
import sys, json, itertools
from fractions import Fraction as F
def pats(K): return [v for v in itertools.product((0, 1), repeat=K) if any(v)]
def reach(n, D, P):
    pts = {tuple([0] * len(P[0]))}
    for c, v in zip(n, P):
        if c:
            pts = {tuple(a + t * x for a, x in zip(p, v)) for p in pts for t in range(c, (D + 1) * c + 1)}
    return pts
tot = bad = 0
for fn in sys.argv[1:]:
    for line in open(fn):
        c = json.loads(line); tot += 1
        K, M = c["K"], c["M"]; P = pats(K)
        D = 3 * M + 2 if c["D"] == "inf" else c["D"]
        at = [(y, tuple(n), F(w)) for y, n, w in c["atoms"]]
        val = {int(i): F(v) for i, v in c["val"].items()}
        mu = {(n, m): F(x) for n, m, x in c["mu"]}
        R = {n: reach(n, D, P) for _, n, _ in at}
        adj = {(i, j) for i, (y, n, _) in enumerate(at) if y == 0 for j, (z, m, _) in enumerate(at)
               if z == 1 and not R[n].isdisjoint(R[m])}
        ok = all(val[j] <= val[i] for i, j in adj)
        ok &= all(x >= 0 and (i, j) in adj and val[i] == val[j] for (i, j), x in mu.items())
        for i, (y, n, w) in enumerate(at):
            s = sum(x for (a, b), x in mu.items() if (a if y == 0 else b) == i)
            ok &= (w * val[i] == s) if y == 0 else (w * (1 - val[i]) == s)
        V = sum(w * (val[i] ** 2 if y == 0 else (1 - val[i]) ** 2) for i, (y, n, w) in enumerate(at))
        ok &= (V == F(c["value"]))
        if not ok:
            bad += 1; print("FAILED", fn, c["kind"], c["trial"], c["D"])
print(f"certificates checked: {tot}; failed: {bad}")
