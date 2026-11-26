"""Whole-report copying (Proposition 3(i), Theorem 3, Proposition 8): degenerate and boundary cases (pytest)."""
from fractions import Fraction as F
from itertools import combinations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "whole_report_saturation"))
from finite_cap import Game, k2_rho, dsat_direct, criterion_dsat
from exact_core import solve_law, patterns

def dstar(K, M):
    g = Game(K, M, 3 * M + 2)
    t = [g.thr(n, m) for n, m in combinations(g.S, 2) if g.thr(n, m) is not None]
    return max(t, default=0)

def test_dstar_k2_small_M():
    assert dstar(2, 1) == 0          # boundary: 2M-3 does not apply at M=1
    for M in (2, 3, 4, 5):
        assert dstar(2, M) == 2 * M - 3

def test_dstar_k1_and_k3():
    for M in range(1, 7):
        assert dstar(1, M) == max(M - 1, 0)
    assert dstar(3, 1) == 0 and dstar(3, 2) == 1 and dstar(3, 3) == 4

def test_zero_mass_entries_ignored():
    g = Game(1, 3, 3)
    law = {(0, (1,)): F(1, 2), (1, (3,)): F(1, 2), (1, (2,)): F(0)}
    law2 = {k: v for k, v in law.items() if v}
    assert solve_law(law, g.edge_D(1))[0] == solve_law(law2, g.edge_D(1))[0]

def test_single_class_and_pure_blocks():
    g = Game(2, 2, 8)
    law = {(1, (0, 1, 0)): F(1, 3), (1, (1, 1, 0)): F(2, 3)}
    assert solve_law(law, g.inf_edge)[0] == 0
    assert dsat_direct(g, law, 8)[0] == 0 and criterion_dsat(g, law, 8)[0] == 0

def test_state_with_both_labels_and_empty_state():
    g = Game(2, 2, 8)
    z = (0, 0, 0)
    law = {(0, z): F(1, 4), (1, z): F(1, 4), (0, (1, 0, 0)): F(1, 2)}
    V = solve_law(law, g.inf_edge)[0]
    assert V == F(1, 2) * F(1, 2) * F(1, 2)   # block {z}: p=1/2, theta=1/2; the other atom is pure
    assert dsat_direct(g, law, 8)[0] == 0

def test_equal_values_in_separate_blocks():
    g = Game(1, 4, 4)
    law = {(0, (1,)): F(1, 8), (1, (1,)): F(1, 8), (0, (0,)): F(1, 8), (1, (0,)): F(1, 8),
           (0, (4,)): F(1, 4), (1, (4,)): F(1, 4)}
    V, val, blocks, atoms, st, below = solve_law(law, g.inf_edge)
    assert V == F(1, 4)
    assert dsat_direct(g, law, 4)[0] == criterion_dsat(g, law, 4)[0]

def test_manuscript_k1_examples():
    g = Game(1, 3, 3)
    ex1 = {(0, (1,)): F(1, 4), (0, (3,)): F(1, 4), (1, (2,)): F(1, 2)}
    assert solve_law(ex1, g.edge_D(1))[0] == F(1, 4)           # Theorem 2 example
    ex2 = {(0, (1,)): F(1, 2), (1, (1,)): F(1, 4), (1, (3,)): F(1, 4)}
    assert solve_law(ex2, g.edge_D(1))[0] == F(1, 6)           # Prop. 4 counterexample
    assert dsat_direct(g, ex2, 3)[0] == 2

def test_common_point_counterexample():
    g = Game(2, 3, 11)
    law = {(1, (1, 2, 0)): F(6, 25), (0, (1, 2, 0)): F(6, 25), (0, (0, 0, 1)): F(8, 25), (0, (1, 0, 1)): F(5, 25)}
    assert solve_law(law, g.inf_edge)[0] == F(114, 625)
    assert solve_law(law, g.edge_D(1))[0] == F(21, 125)
    assert dsat_direct(g, law, 11)[0] == 2 == criterion_dsat(g, law, 11)[0]

def test_k2_formula_at_M1():
    g = Game(2, 1, 5)
    for n, m in combinations(g.S, 2):
        rho = k2_rho(n, m)
        assert (rho is None) == (g.thr(n, m) is None)

def test_gap_bound_and_quadratic_counterexample():
    from quant_gap import blocks_inf, delta
    from exact_core import build_atoms, below_from_edges, isotonic, value
    g = Game(2, 3, 11)
    law = {(0, (0, 0, 1)): F(4, 9), (0, (1, 1, 1)): F(1, 3), (1, (1, 0, 1)): F(1, 9), (1, (1, 1, 1)): F(1, 9)}
    atoms, st, vinf, B = blocks_inf(law, g)
    a2, s2 = build_atoms(law); v0, _ = isotonic(a2, below_from_edges(a2, s2, g.edge_D(0)))
    G = value(atoms, vinf) - value(a2, v0)
    UB = LB = F(0)
    for th, S in B.items():
        d, p0, p1 = delta(atoms, st, S, g.edge_D(0)); h = p0 * p1 / (p0 + p1)
        UB += h * d; LB += h * 4 * th * (1 - th) * d * d
    assert len(B) == 2 and G == F(4, 45) and UB == F(31, 180) and LB == F(2149, 18000)
    assert LB > G and G <= UB
