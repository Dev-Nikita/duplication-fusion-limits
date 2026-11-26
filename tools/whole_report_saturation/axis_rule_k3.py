"""K=3: separating-axis rule with the 9 plane normals (cross products of pattern pairs) vs exact
lattice enumeration of reachable sets. Exact integer arithmetic. Usage: python axis_rule_k3.py"""
import sys, itertools
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from exact_core import patterns, states, reach
from signature_cells import cross, prim
P = patterns(3)
normals = sorted({prim(cross(u, v)) for u, v in itertools.combinations(P, 2) if any(cross(u, v))})
def rng(n, D, c):
    lo = hi = 0
    for k, v in zip(n, P):
        cv = sum(a * b for a, b in zip(c, v))
        a, b = k * cv, (D + 1) * k * cv
        lo += min(a, b); hi += max(a, b)
    return lo, hi
def axis_meet(n, m, D):
    for c in normals:
        a, b = rng(n, D, c); x, y = rng(m, D, c)
        if b < x or y < a:
            return False
    return True
tot = bad = 0
for M in (1, 2, 3):
    S = states(3, M)
    for D in range(0, 3 * M + 1):
        R = {n: reach(n, D, P) for n in S}
        for n, m in itertools.combinations(S, 2):
            tot += 1
            if axis_meet(n, m, D) != (not R[n].isdisjoint(R[m])):
                bad += 1
                if bad < 5: print("mismatch", M, D, n, m)
msg = f"K=3 nine-axis rule vs exact lattice enumeration: {tot} (M,D,pair) cases, M<=3, D<=3M: {bad} mismatches; normals={normals}"
print(msg); open("out/k3_axis_rule.txt", "w").write(msg + "\n")
