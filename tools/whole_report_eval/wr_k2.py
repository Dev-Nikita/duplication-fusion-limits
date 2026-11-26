"""Phase C library for two report types (K=2): whole-report vs coordinate-wise copying models.

State n = (n01, n10, n11): number of sources whose whole report has pattern (0,1), (1,0), (1,1)  [exact_core order].
Observation A = (A0, A1) = (t10 + t11, t01 + t11), t_v copies of pattern v, n_v <= t_v <= (D+1) n_v (D=None: no cap).
Model classes for the copying mechanism (all with cap D):
  whole      : t_v chosen per pattern (whole reports are copied)                        -> reach set R_D(n)
  coord      : A_k in [U_k, (D+1) U_k] independently, U = (n10+n11, n01+n11)           -> box B_D(n)
  pattern    : the fuser sees the copied pattern counts t itself                       -> product of intervals
Mechanism information: 'label' (kernel may depend on (n, y)) or 'state' (kernel depends on n only, label-blind).

Minimax values for the label-aware class are exact (rational isotonic regression, exact_core). A minimax rule is
built from the optimal levels: f(a) = min{u_n : y=0 atom n with a in R(n)}, else max{l_m : y=1 atom with a in R(m)},
else the fallback rule (fixed before any test data: the smoothed coordinate-presence posterior of the training law).
Worst cases of a fixed rule are computed on the integer grid; they are exact for D < infinity.
"""
import math, os, sys
from fractions import Fraction as F
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "whole_report_saturation"))
from exact_core import build_atoms, below_from_edges, isotonic, certify, value, Flow  # noqa: E402

INF = None


def _ratio(a, b):
    if a == 0 and b == 0:
        return F(1)
    if a == 0 or b == 0:
        return None
    return max(F(a, b), F(b, a))


def _thr(rs):
    if any(r is None for r in rs):
        return None
    return max(0, math.ceil(max(rs)) - 1)


def thr_whole(n, m):
    """least cap D with R_D(n) and R_D(m) intersecting (K=2 closed form, Theorem 3 / Table S15); None if never."""
    n01, n10, n11 = n; m01, m10, m11 = m
    return _thr([_ratio(n10 + n11, m10 + m11), _ratio(n01 + n11, m01 + m11), _ratio(n10 + m01, m10 + n01)])


def marg(n):
    return (n[1] + n[2], n[0] + n[2])


def thr_coord(n, m):
    (a0, a1), (b0, b1) = marg(n), marg(m)
    return _thr([_ratio(a0, b0), _ratio(a1, b1)])


def thr_pattern(n, m):
    return _thr([_ratio(x, y) for x, y in zip(n, m)])


THR = {"whole": thr_whole, "coord": thr_coord, "pattern": thr_pattern}


def edge(model, D):
    t = THR[model]
    if D is INF:
        return lambda n, m: t(n, m) is not None
    return lambda n, m: (t(n, m) is not None and t(n, m) <= D)


# ------------------------------------------------------------------ membership on the observation grid
def _iv(c, D):
    lo = c
    hi = (math.inf if c > 0 else 0) if D is INF else (D + 1) * c
    return lo, hi


def member(model, n, D, a0, a1):
    """boolean array: is the observation (a0, a1) reachable from state n under the model class (marginal-count
    observation; for 'pattern' use member_pattern)."""
    if model == "coord":
        (u0, u1) = marg(n)
        l0, h0 = _iv(u0, D); l1, h1 = _iv(u1, D)
        return (a0 >= l0) & (a0 <= h0) & (a1 >= l1) & (a1 <= h1)
    assert model == "whole"
    n01, n10, n11 = n
    lt, ht = _iv(n11, D); l10, h10 = _iv(n10, D); l01, h01 = _iv(n01, D)
    lo = np.maximum(np.maximum(lt, a0 - h10), a1 - h01)
    hi = np.minimum(np.minimum(ht, a0 - l10), a1 - l01)
    return lo <= hi


def grid(M, D):
    top = (D + 1) * M if D is not INF else 4 * M  # D=inf: rules are only evaluated, never maximised, beyond this box
    a0, a1 = np.meshgrid(np.arange(top + 1), np.arange(top + 1), indexing="ij")
    return a0.ravel(), a1.ravel()


# ------------------------------------------------------------------ laws
def law_from_counts(cnt):
    """cnt: dict (y, n) -> int. Returns rational law."""
    tot = sum(cnt.values())
    return {k: F(v, tot) for k, v in cnt.items() if v > 0}


def bayes(pairs):
    """Bayes Brier risk of a channel given an iterable of (p0, p1) masses per observation."""
    return sum(F(p0) * F(p1) / (F(p0) + F(p1)) for p0, p1 in pairs if p0 + p1 > 0)


def channel_law(law, fn):
    out = {}
    for (y, n), w in law.items():
        k = fn(n); a, b = out.get(k, (F(0), F(0)))
        out[k] = (a + w, b) if y == 0 else (a, b + w)
    return out


def presence(n):
    u = marg(n); return (int(u[0] > 0), int(u[1] > 0))


# ------------------------------------------------------------------ exact minimax (label-aware)
def minimax(law, model, D):
    atoms, st = build_atoms(law)
    below = below_from_edges(atoms, st, edge(model, D))
    val, blocks = isotonic(atoms, below)
    assert certify(atoms, below, val)
    return value(atoms, val), val, atoms, st


class MinimaxRule:
    def __init__(self, law, model, D, fallback):
        self.V, val, atoms, st = minimax(law, model, D)
        self.model, self.D, self.fallback = model, D, fallback
        self.up = [(st[i], float(val[i])) for i, (y, _) in enumerate(atoms) if y == 0]
        self.lo = [(st[i], float(val[i])) for i, (y, _) in enumerate(atoms) if y == 1]

    def covered(self, a0, a1):
        c = np.zeros(np.shape(a0), bool)
        for n, _ in self.up + self.lo:
            c |= member(self.model, n, self.D, a0, a1)
        return c

    def __call__(self, a0, a1):
        u = np.full(a0.shape, np.inf); l = np.full(a0.shape, -np.inf)
        for n, v in self.up:
            m = member(self.model, n, self.D, a0, a1); u[m] = np.minimum(u[m], v)
        for n, v in self.lo:
            m = member(self.model, n, self.D, a0, a1); l[m] = np.maximum(l[m], v)
        out = np.where(np.isfinite(u), u, np.where(np.isfinite(l), l, np.nan))
        fb = np.isnan(out)
        out[fb] = self.fallback(a0[fb], a1[fb])
        return out


# ------------------------------------------------------------------ baselines trained on observed (a0, a1)
class PresenceRule:
    def __init__(self, obs):
        """obs: list of (y, a0, a1) training observations."""
        c = {}
        for y, a0, a1 in obs:
            k = (a0 > 0, a1 > 0); n0, n1 = c.get(k, (0, 0)); c[k] = (n0 + (y == 0), n1 + (y == 1))
        prior = np.mean([y for y, _, _ in obs])
        self.t = {k: (n1 + prior) / (n0 + n1 + 1) for k, (n0, n1) in c.items()}; self.prior = prior

    def __call__(self, a0, a1):
        return np.array([self.t.get((x > 0, z > 0), self.prior) for x, z in zip(a0, a1)], dtype=float)


class CountTable:
    """smoothed empirical Bayes on the observed count pair, backing off to presence (strength k=1)."""
    def __init__(self, obs, back):
        c = {}
        for y, a0, a1 in obs:
            n0, n1 = c.get((a0, a1), (0, 0)); c[(a0, a1)] = (n0 + (y == 0), n1 + (y == 1))
        self.c, self.back = c, back

    def __call__(self, a0, a1):
        b = self.back(a0, a1); out = b.copy()
        for i, (x, z) in enumerate(zip(a0, a1)):
            if (x, z) in self.c:
                n0, n1 = self.c[(x, z)]; out[i] = (n1 + b[i]) / (n0 + n1 + 1)
        return out


class Logistic:
    """L2-regularised logistic regression on transformed counts with interaction; kind in {'log', 'clip'}."""
    def __init__(self, obs, kind, M, lam=1e-2):
        from scipy.optimize import minimize
        self.kind, self.M = kind, M
        y = np.array([o[0] for o in obs], float); X = self._X(np.array([o[1] for o in obs]), np.array([o[2] for o in obs]))
        def nll(w):
            z = X @ w; return np.sum(np.logaddexp(0, z) - y * z) / len(y) + lam * np.sum(w[1:] ** 2)
        self.w = minimize(nll, np.zeros(X.shape[1]), method="L-BFGS-B").x

    def _X(self, a0, a1):
        a0 = np.asarray(a0, float); a1 = np.asarray(a1, float)
        if self.kind == "clip":
            f0, f1 = np.minimum(a0, self.M) / self.M, np.minimum(a1, self.M) / self.M
        else:
            f0, f1 = np.log1p(a0), np.log1p(a1)
        return np.column_stack([np.ones_like(f0), f0, f1, f0 * f1])

    def __call__(self, a0, a1):
        return 1 / (1 + np.exp(-(self._X(a0, a1) @ self.w)))


# ------------------------------------------------------------------ worst cases of a fixed rule
def worst_case(rule, law, model, D, info, M):
    """exact worst-case Brier risk of a fixed rule on 'law' over the mechanism class (model, cap D, information info)."""
    assert D is not INF
    a0, a1 = grid(M, D); f = rule(a0, a1)
    by_n = {}
    for (y, n), w in law.items():
        p = by_n.setdefault(n, [0.0, 0.0]); p[y] += float(w)
    r = 0.0
    for n, (p0, p1) in by_n.items():
        m = member(model, n, D, a0, a1); fn = f[m]
        if info == "label":
            r += p0 * np.max(fn ** 2) + p1 * np.max((1 - fn) ** 2)
        else:
            r += np.max(p0 * fn ** 2 + p1 * (1 - fn) ** 2)
    return r


def observed_risk(rule, law):
    """Brier risk without copying (A = marginal counts)."""
    r = 0.0
    for (y, n), w in law.items():
        u = marg(n); f = rule(np.array([u[0]]), np.array([u[1]]))[0]; r += float(w) * (f - y) ** 2
    return r


# ------------------------------------------------------------------ saturation gap and certified bound (Prop. 8)
def gap_bound(law, D):
    """G_D = R*_inf - R*_D (whole) and B_D = sum_S h_S delta_{D,S} over mixed blocks of the unrestricted optimum."""
    Vinf, vinf, atoms, st = minimax(law, "whole", INF)
    VD = minimax(law, "whole", D)[0]
    lev = {}
    for i, v in vinf.items():
        lev.setdefault(v, []).append(i)
    B = F(0); e = edge("whole", D)
    for th, S in lev.items():
        if not (0 < th < 1):
            continue
        z = [i for i in S if atoms[i][0] == 0]; o = [i for i in S if atoms[i][0] == 1]
        p0 = sum(atoms[i][1] for i in z); p1 = sum(atoms[i][1] for i in o)
        idx = {a: k + 2 for k, a in enumerate(S)}; fl = Flow(len(S) + 2)
        for i in z:
            fl.add(0, idx[i], atoms[i][1] / p0)
        for j in o:
            fl.add(idx[j], 1, atoms[j][1] / p1)
        for i in z:
            for j in o:
                if e(st[i], st[j]):
                    fl.add(idx[i], idx[j], F(2))
        B += (p0 * p1) / (p0 + p1) * (1 - fl.maxflow(0, 1))
    return Vinf - VD, B, Vinf, VD


# ------------------------------------------------------------------ observation channels
def channel_table(law, D, M):
    """robust (worst-case over the whole-report class, label-aware) Brier risk of each observation channel, and the
    bits per episode the fuser needs. Copy-invariant channels have robust risk = their Bayes risk."""
    bits_cnt = math.ceil(math.log2((D + 1) * M + 1))
    rows = []
    rows.append(("coordinate presence", float(bayes(channel_law(law, presence).values())), 2, "none"))
    rows.append(("pattern presence", float(bayes(channel_law(law, lambda n: tuple(int(c > 0) for c in n)).values())), 3, "none"))
    rows.append(("marginal counts", float(minimax(law, "whole", D)[0]), 2 * bits_cnt, "none"))
    rows.append(("pattern counts", float(minimax(law, "pattern", D)[0]), 3 * bits_cnt, "pattern kept per message"))
    rows.append(("unique marginal counts (negatives)", float(bayes(channel_law(law, marg).values())),
                 2 * math.ceil(math.log2(M + 1)), "complete negative reports from a known population"))
    rows.append(("unique-source patterns (lineage)", float(bayes(channel_law(law, lambda n: n).values())),
                 3 * math.ceil(math.log2(M + 1)), "source identity per message"))
    return rows
