"""Query policies over a HypTable. Policies see the table and the answers so far — never the truth."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from paef.g3.hypotheses import HypTable, status


def _answers(t: HypTable, mask: np.ndarray, k: int):
    vals = np.unique(t.codes[mask, k])
    out = []
    tot = t.w[mask].sum()
    for v in vals:
        m2 = mask & (t.codes[:, k] == v)
        out.append((v, t.w[m2].sum() / tot if tot > 0 else 1.0 / len(vals), m2))
    return out


def _open(t, mask, asked):
    return [k for k in range(len(t.nodes)) if k not in asked and len(np.unique(t.codes[mask, k])) > 1]


def random_policy(t, mask, asked, rng):
    c = _open(t, mask, asked)
    return int(rng.choice(c)) if c else None


def max_uncertainty(t, mask, asked, rng):
    best, bk = -1, None
    for k in _open(t, mask, asked):
        p = np.array([a[1] for a in _answers(t, mask, k)])
        h = -(p[p > 0] * np.log(p[p > 0])).sum()
        if h > best:
            best, bk = h, k
    return bk


def expected_interval_reduction(t, mask, asked, rng):
    best, bk = np.inf, None
    for k in _open(t, mask, asked):
        ew = sum(p * (t.hi[m].max() - t.lo[m].min()) for _, p, m in _answers(t, mask, k) if m.any())
        if ew < best:
            best, bk = ew, k
    return bk


def decision_robustness(t, mask, asked, rng):
    best, bk = -1, None
    for k in _open(t, mask, asked):
        pd = sum(p for _, p, m in _answers(t, mask, k) if status(t, m) != -1)
        # tie-break by expected width
        ew = sum(p * (t.hi[m].max() - t.lo[m].min()) for _, p, m in _answers(t, mask, k) if m.any())
        score = pd - 1e-6 * ew
        if score > best:
            best, bk = score, k
    return bk


def _region(t):
    return np.where(t.lo > t.tau_lo, 1, np.where(t.hi < t.tau_lo, 0, 2))


def ec2(t, mask, asked, rng):
    reg = _region(t)

    def edge_weight(m):
        tot = 0.0
        ws = [t.w[m & (reg == r)].sum() for r in (0, 1, 2)]
        for a in range(3):
            for b in range(a + 1, 3):
                tot += ws[a] * ws[b]
        return tot

    base = edge_weight(mask)
    best, bk = -1, None
    for k in _open(t, mask, asked):
        # remaining edge weight after the query = sum over answers of the weight inside each answer set
        rem = sum(edge_weight(m) for _, p, m in _answers(t, mask, k))
        cut = base - rem
        if cut > best:
            best, bk = cut, k
    return bk


def dp_optimal_value(t: HypTable, max_nodes: int = 5) -> Optional[float]:
    """Minimum expected number of queries to a robust decision (or exhaustion) under prior w."""
    n = len(t.nodes)
    if n > max_nodes:
        return None

    @lru_cache(maxsize=None)
    def V(key):
        mask = np.frombuffer(key, dtype=bool)
        if status(t, mask) != -1:
            return 0.0
        opts = _open(t, mask, set())
        if not opts:
            return 0.0
        best = np.inf
        for k in opts:
            v = 1.0 + sum(p * V(m.tobytes()) for _, p, m in _answers(t, mask, k))
            best = min(best, v)
        return best

    return float(V(np.ones(len(t.lo), dtype=bool).tobytes()))


POLICIES = {"random": random_policy, "max_uncertainty": max_uncertainty,
            "expected_interval_reduction": expected_interval_reduction,
            "decision_robustness": decision_robustness, "ec2": ec2}


def run_policy(t: HypTable, truth_codes: np.ndarray, policy, rng) -> Dict:
    """Simulate a policy; `truth_codes` is consulted only to answer the chosen query (the oracle)."""
    mask = np.ones(len(t.lo), dtype=bool)
    asked: List[int] = []
    while status(t, mask) == -1:
        k = policy(t, mask, set(asked), rng)
        if k is None:
            break
        asked.append(k)
        mask = mask & (t.codes[:, k] == truth_codes[k])
        if not mask.any():  # truth outside calibrated sets: certificate failed
            return {"queries": len(asked), "decision": -1, "set_miss": True}
    return {"queries": len(asked), "decision": status(t, mask), "set_miss": False}
