import os, sys, random
from fractions import Fraction as F
from itertools import combinations
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "whole_report_eval"))
sys.path.insert(0, os.path.join(ROOT, "tools", "whole_report_saturation"))
import wr_k2 as W  # noqa: E402
from exact_core import states, patterns, reach, build_atoms, below_from_edges, isotonic, value  # noqa: E402


def rand_law(M, rnd, size=8):
    S = [s for s in states(2, M) if any(s)]
    cnt = {}
    for _ in range(size):
        k = (rnd.randint(0, 1), rnd.choice(S)); cnt[k] = cnt.get(k, 0) + rnd.randint(1, 5)
    if len({y for y, _ in cnt}) < 2:
        cnt[(1 - next(iter(cnt))[0], rnd.choice(S))] = 1
    return W.law_from_counts(cnt)


def test_threshold_matches_enumeration():
    P = patterns(2)
    for M in (1, 2, 3):
        for n, m in combinations(states(2, M), 2):
            t = None
            for D in range(3 * M + 3):
                if not reach(n, D, P).isdisjoint(reach(m, D, P)):
                    t = D; break
            assert W.thr_whole(n, m) == t


def test_membership_matches_enumeration():
    P = patterns(2)
    for n in states(2, 3):
        for D in (0, 1, 2):
            a0, a1 = W.grid(3, D); mem = W.member("whole", n, D, a0, a1)
            R = reach(n, D, P)
            assert {(int(x), int(z)) for x, z, k in zip(a0, a1, mem) if k} == set(R)


def test_rule_attains_value_and_orderings():
    rnd = random.Random(7)
    for trial in range(40):
        M = rnd.randint(1, 4); law = rand_law(M, rnd); D = rnd.randint(0, 3)
        fb = lambda a0, a1: np.full(np.shape(a0), 0.5)
        r = W.MinimaxRule(law, "whole", D, fb)
        wc = W.worst_case(r, law, "whole", D, "label", M)
        assert abs(wc - float(r.V)) < 1e-9
        assert W.worst_case(r, law, "whole", D, "state", M) <= wc + 1e-12
        rc = W.MinimaxRule(law, "coord", D, fb)
        assert rc.V >= r.V                                   # coordinate class contains the whole-report class
        assert W.worst_case(rc, law, "whole", D, "label", M) >= float(r.V) - 1e-12
        G, B, Vinf, VD = W.gap_bound(law, D)
        assert 0 <= G <= B


def test_coordinate_value_matches_bruteforce_edges():
    rnd = random.Random(3)
    for trial in range(20):
        M = rnd.randint(1, 3); law = rand_law(M, rnd); D = rnd.randint(0, 2)
        def box(n):
            u = W.marg(n); return {(a, b) for a in range(u[0], (D + 1) * u[0] + 1) for b in range(u[1], (D + 1) * u[1] + 1)}
        atoms, st = build_atoms(law)
        val, _ = isotonic(atoms, below_from_edges(atoms, st, lambda n, m: not box(n).isdisjoint(box(m))))
        assert value(atoms, val) == W.minimax(law, "coord", D)[0]
