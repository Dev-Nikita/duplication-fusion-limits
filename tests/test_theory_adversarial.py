"""Adversarial and edge-case checks of the finite-cap theory (Theorem 2, Propositions 4-7).
They test implementations and agreement between independent routes; they do not replace the proofs."""
import itertools, math, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
cp = pytest.importorskip("cvxpy")
import finite_cap_cvx as C, finite_cap_solvers as S, saturation_closed_form as SC, quantitative_saturation as QS, extended_models as E  # noqa: E402


def rand_law(rng, M, zero_mass=True):
    p = rng.dirichlet(np.ones(2 * (M + 1))).reshape(2, M + 1)
    if not zero_mass:
        p[:, 0] = 0; p /= p.sum()
    return p


def test_exhaustive_grid_brackets_theorem2():
    """M = 2, D = 1: exhaustive grid over all rules x in [0,1]^5 (no convex solver) brackets the program's value."""
    rng = np.random.default_rng(1); g = np.linspace(0, 1, 21)
    X = np.array(list(itertools.product(g, repeat=5)))          # x_0..x_4
    for _ in range(5):
        p = rand_law(rng, 2); I = S.intervals(2, 1)
        risk = sum(p[0, u] * (X[:, I[u]] ** 2).max(1) + p[1, u] * ((1 - X[:, I[u]]) ** 2).max(1) for u in range(3))
        v = C.ld_minimax(p, 1)[0]
        assert risk.min() >= v - 1e-6                             # every rule is an upper bound
        assert risk.min() <= v + 0.01                             # the grid gets close


def test_deterministic_kernels_are_lower_bounds_and_can_be_strict():
    """Bayes risk of any deterministic kernel <= R*_D; the manuscript's example needs a randomised kernel."""
    p = np.zeros((2, 4)); p[0, 1] = p[0, 3] = 0.25; p[1, 2] = 0.5; I = S.intervals(3, 1); best = 0.0
    for a01, a03, a12 in itertools.product(I[1], I[3], I[2]):
        Q0 = [np.eye(len(I[u]))[0] for u in range(4)]; Q1 = [np.eye(len(I[u]))[0] for u in range(4)]
        Q0[1] = (I[1] == a01).astype(float); Q0[3] = (I[3] == a03).astype(float); Q1[2] = (I[2] == a12).astype(float)
        best = max(best, S.dual_value(Q0, Q1, p, 1)[0])
    v = C.ld_minimax(p, 1)[0]
    assert abs(v - 0.25) < 1e-5 and best <= v + 1e-9 and best < 0.25 - 1e-3


@pytest.mark.parametrize("M", [1, 2, 5])
def test_edge_caps(M):
    rng = np.random.default_rng(M); p = rand_law(rng, M)
    assert abs(C.ld_minimax(p, 0)[0] - S.bayes_risk_U(p)) < 1e-6                      # D = 0
    for D in (max(M - 1, 0), M + 3):                                                   # D >= M-1
        assert abs(C.ld_minimax(p, D)[0] - S.bayes_risk_presence(p)) < 1e-6
        assert abs(C.li_minimax(p, D)[0] - S.bayes_risk_presence(p)) < 1e-5            # padding is count-only


def test_degenerate_priors_and_empty_classes():
    p = np.zeros((2, 4)); p[0, 0] = 0.3; p[0, 2] = 0.7                                 # only y = 0
    assert abs(C.ld_minimax(p, 1)[0]) < 1e-7 and abs(S.bayes_risk_presence(p)) < 1e-12
    p = np.zeros((2, 4)); p[0, 0] = 0.5; p[1, 2] = 0.5                                 # class 0 never fires: presence is sufficient
    assert abs(C.ld_minimax(p, 1)[0]) < 1e-7
    p = np.zeros((2, 4)); p[0, 0] = 0.25; p[1, 0] = 0.25; p[0, 1] = 0.25; p[1, 1] = 0.25   # U carries nothing
    for D in (0, 1, 2):
        assert abs(C.ld_minimax(p, D)[0] - 0.25) < 1e-6


def test_saturation_closed_form_vs_lp_and_value():
    rng = np.random.default_rng(7)
    for _ in range(40):
        M = int(rng.integers(2, 8)); p = rand_law(rng, M)
        p[:, rng.integers(1, M + 1)] = 0; p /= p.sum()                                 # holes in the support
        if p[0, 1:].sum() == 0 or p[1, 1:].sum() == 0:
            continue
        dsat = math.ceil(SC.rho(p) - 1e-12) - 1; assert 0 <= dsat <= M - 1
        for D in range(M):
            sat = D >= dsat
            assert SC.lp_saturated(p, D) == sat == SC.closed_form_saturated(p, D)
            gap = S.bayes_risk_presence(p) - C.ld_minimax(p, D)[0]
            assert (gap < 1e-6) == sat or abs(gap) < 1e-5


def test_overlapping_supports_do_not_imply_saturation():
    p = np.zeros((2, 4)); p[0, 1] = 0.5; p[1, 1] = 0.25; p[1, 3] = 0.25
    assert not SC.lp_saturated(p, 1) and SC.lp_saturated(p, 2)
    assert abs(C.ld_minimax(p, 1)[0] - 1 / 6) < 1e-5


def test_value_of_counts_bounds_unequal_masses():
    rng = np.random.default_rng(11)
    for _ in range(40):
        M = int(rng.integers(2, 8)); p = rand_law(rng, M) * np.array([[rng.uniform(0.2, 3)], [1.0]]); p /= p.sum()
        lo_c, h = QS.constants(p)
        for D in range(M):
            V = S.bayes_risk_presence(p) - C.ld_minimax(p, D)[0]; d = QS.delta(p, D)
            assert lo_c * d * d - 1e-6 <= V <= h * d + 1e-6 and d <= QS.beta(p, D) + 1e-9


def test_mechanism_information_ordering_and_equalities():
    rng = np.random.default_rng(3); M, D = 3, 1
    sets = [s for r in range(1, M + 1) for s in itertools.combinations(range(M), r)]
    for trial in range(15):
        w = rng.dirichlet(np.ones(2 * len(sets))).reshape(2, len(sets))
        if trial % 3 == 1:                                # exchangeable: mass depends on |S| only
            base = rng.dirichlet(np.ones(2 * M)).reshape(2, M)
            w = np.array([[base[y, len(s) - 1] / math.comb(M, len(s)) for s in sets] for y in (0, 1)])
        if trial % 3 == 2:                                # S determines y
            mask = rng.integers(0, 2, len(sets)); w[0] *= mask; w[1] *= 1 - mask; w /= w.sum()
        pS = {s: (w[0, i], w[1, i]) for i, s in enumerate(sets)}
        p = np.zeros((2, M + 1))
        for s, (a, b) in pS.items():
            p[0, len(s)] += a; p[1, len(s)] += b
        v_u = E.solve_cells([(E.interval(u, D), p[0, u], p[1, u]) for u in range(M + 1) if p[0, u] + p[1, u] > 0])[0]
        v_s = E.solve_cells([(E.interval(len(s), D), a, b) for s, (a, b) in pS.items()])[0]
        v_y = C.ld_minimax(p, D)[0]
        assert S.bayes_risk_U(p) - 1e-6 <= v_u <= v_s + 1e-6 <= v_y + 2e-6 <= S.bayes_risk_presence(p) + 3e-6
        if trial % 3 == 1:
            assert abs(v_s - v_u) < 1e-5
        if trial % 3 == 2:
            assert abs(v_s - v_y) < 1e-5


def test_two_type_example():
    p = {(1, 3): (0.25, 0.0), (3, 1): (0.25, 0.0), (1, 1): (0.0, 0.25), (3, 3): (0.0, 0.25)}
    assert abs(E.solve_cells(E.cells_label_aware(p, 1))[0]) < 1e-6
    assert abs(E.solve_cells(E.cells_label_aware(p, 2))[0] - 0.25) < 1e-6
