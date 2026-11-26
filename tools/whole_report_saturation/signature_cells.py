"""G4: exact, provably exhaustive enumeration of conic signatures for K=2 and K=3.

Signature of a nonzero a in R^K_{>=0}: sig(a) = {V nonempty subset of patterns : a in relint cone(V)}.
Every relint cone(V) is a union of relatively open cells of the central arrangement of the
hyperplanes spanned by K-1 independent patterns (for K=3: planes through pairs of patterns,
including the coordinate planes). Hence sig is constant on each cell, and enumerating one
rational point per cell is exhaustive. On the simplex sum(a)=1 the arrangement is a line
arrangement (K=3) or a point set (K=2); the representatives are: every vertex, the midpoint of
every pair of consecutive vertices on every line, and the two points q +- eps*d next to every such
midpoint, with eps computed exactly so that no other line is crossed. Every 2-cell has a
boundary edge, so it receives a representative. Membership tests are exact (rational witness or
integer Farkas certificate, see exact_core.relint_meet).
"""
import sys, csv, time, itertools
from fractions import Fraction as F
from math import gcd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from exact_core import patterns, relint_meet

import os
OUT = os.path.join(os.getcwd(), "out")  # run from results/whole_report


def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def prim(c):
    g = 0
    for x in c:
        g = gcd(g, abs(x))
    c = tuple(x // g for x in c)
    for x in c:
        if x != 0:
            return c if x > 0 else tuple(-y for y in c)
    return c


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def reps_K3():
    P = patterns(3)
    lines = set()
    for u, v in itertools.combinations(P, 2):
        c = cross(u, v)
        if any(c):
            lines.add(prim(c))
    for k in range(3):
        lines.add(tuple(1 if i == k else 0 for i in range(3)))
    lines = sorted(lines)
    verts = set()
    for c1, c2 in itertools.combinations(lines, 2):
        d = cross(c1, c2)
        s = sum(d)
        if s == 0:
            continue
        a = tuple(F(x, s) for x in d)
        if all(x >= 0 for x in a):
            verts.add(a)
    pts = {("vertex", v) for v in verts}
    for c in lines:
        on = [v for v in verts if dot(c, v) == 0]
        # parametrize along the line by a direction inside the simplex plane
        dvec = cross(c, (1, 1, 1))
        on.sort(key=lambda v: dot(dvec, v))
        for v, w in zip(on, on[1:]):
            q = tuple((x + y) / 2 for x, y in zip(v, w))
            pts.add(("edge", q))
            m = F(sum(c), 3)
            d = tuple(F(x) - m for x in c)  # sum(d)=0, stays in the simplex plane
            if dot(c, d) == 0:
                raise RuntimeError("degenerate normal")
            eps = None
            for c2 in lines:
                cd, cq = dot(c2, d), dot(c2, q)
                if cd != 0 and cq != 0:
                    r = abs(cq) / abs(cd)
                    eps = r if eps is None else min(eps, r)
            eps = eps / 2
            for sgn in (1, -1):
                p = tuple(x + sgn * eps * y for x, y in zip(q, d))
                if all(x >= 0 for x in p):
                    pts.add(("face", p))
    return P, lines, sorted(verts), pts


def reps_K2():
    P = patterns(2)
    # rays of the patterns on the segment a1+a2=1: (0,1),(1,0),(1/2,1/2)
    verts = sorted({(F(v[0], sum(v)), F(v[1], sum(v))) for v in P})
    pts = {("vertex", v) for v in verts}
    for v, w in zip(verts, verts[1:]):
        pts.add(("edge", tuple((x + y) / 2 for x, y in zip(v, w))))
    return P, None, verts, pts


def to_int(a):
    den = 1
    for x in a:
        den = den * x.denominator // gcd(den, x.denominator)
    return tuple(int(x * den) for x in a)


def signature(a, P, subsets):
    ai = to_int(a)
    sig = []
    for V in subsets:
        ok, _ = relint_meet([P[i] for i in V], [ai])
        if ok:
            sig.append(V)
    return frozenset(sig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, summary = [], []
    for K in (2, 3):
        t0 = time.time()
        P, lines, verts, pts = reps_K3() if K == 3 else reps_K2()
        subsets = [V for r in range(1, len(P) + 1) for V in itertools.combinations(range(len(P)), r)]
        cells = {}
        for kind, a in sorted(pts, key=lambda t: (t[0], t[1])):
            s = signature(a, P, subsets)
            cells.setdefault(s, []).append((kind, a))
        # distinct signatures restricted to supports of size <= s
        restricted = {}
        for smax in range(1, len(P) + 1):
            restricted[smax] = len({frozenset(V for V in sig if len(V) <= smax) for sig in cells} - {frozenset()})
        for sig, reps in cells.items():
            kind, a = reps[0]
            minsize = min(len(V) for V in sig) if sig else None
            rows.append(dict(K=K, cell_kind=kind, witness="(" + ",".join(str(x) for x in a) + ")",
                             n_supports=len(sig), min_support_size=minsize,
                             reachable_min_M=minsize,
                             supports=";".join("{" + ",".join("".join(map(str, P[i])) for i in V) + "}" for V in sorted(sig, key=lambda V: (len(V), V)))))
        summary.append(f"K={K}: patterns={len(P)}, lines={0 if lines is None else len(lines)}, vertices={len(verts)}, "
                       f"representatives={len(pts)}, distinct nonzero signatures={len(cells)} "
                       f"(+ the zero signature {{empty support}}); restricted to support size<=s: "
                       + ", ".join(f"s={k}:{v}" for k, v in restricted.items())
                       + f"; seconds={time.time()-t0:.1f}")
        print(summary[-1], flush=True)
    with open(f"{OUT}/signature_catalog.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    open(f"{OUT}/signature_summary.txt", "w").write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
