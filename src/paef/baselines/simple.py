"""Reference fusion rules.

All rules consume only what a deployed system would have: observed values, calibrated
per-observation marginals P(Z_i|H), channel / root parameters, and the *declared*
candidate sets. ``oracle`` alone reads ``true_root`` and is an upper reference, never
presented as a competitor.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from paef.model.lineage import (FRESH, Episode, copy_parent, is_copy,
                                log_odds_given_assignment, marginal_lik, sigmoid)


def _prior_lo(ep: Episode) -> float:
    return float(np.log(ep.prior) - np.log1p(-ep.prior))


def _node_loglr(ep: Episode, i: int) -> float:
    m0, m1 = marginal_lik(ep, i)
    if ep.obs[i].value == 1:
        return float(np.log(m1) - np.log(m0))
    return float(np.log1p(-m1) - np.log1p(-m0))


def naive_bayes(ep: Episode) -> float:
    """Conditional independence given H for every observation."""
    lo = _prior_lo(ep) + sum(_node_loglr(ep, i) for i in ep.observed())
    return float(sigmoid(lo))


def oracle(ep: Episode) -> float:
    """Exact posterior under the TRUE lineage (reference upper bound)."""
    assign = {i: (ep.obs[i].true_code if ep.obs[i].true_code is not None else ep.obs[i].true_root)
              for i in ep.observed()}
    return float(sigmoid(log_odds_given_assignment(ep, assign)))


def declared_only(ep: Episode) -> float:
    """Exact lineage model, but every observation with unresolved provenance is
    treated as an independent origin (absence of a record read as independence)."""
    assign = {}
    for i in ep.observed():
        c = ep.obs[i].candidates
        assign[i] = c[0] if len(c) == 1 else FRESH
    return float(sigmoid(log_odds_given_assignment(ep, assign)))


def _components(ep: Episode, closure: bool) -> List[List[int]]:
    """Countability components.

    closure=False : only declared (singleton) provenance links nodes (PACT as specified,
                    unresolved nodes are separate components).
    closure=True  : nodes that *may* share a root are merged (conservative closure).
    """
    idx = ep.observed()
    parent = {i: i for i in idx}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    by_root: Dict[int, List[int]] = {}
    for i in idx:
        c = ep.obs[i].candidates
        if not closure and len(c) != 1:
            continue
        for r in c:
            if r == FRESH:
                continue
            if is_copy(r):
                p = copy_parent(r)
                if p in parent:
                    ra, rb = find(i), find(p)
                    if ra != rb:
                        parent[ra] = rb
                continue
            by_root.setdefault(r, []).append(i)
    for members in by_root.values():
        for a in members[1:]:
            ra, rb = find(a), find(members[0])
            if ra != rb:
                parent[ra] = rb
    out: Dict[int, List[int]] = {}
    for i in idx:
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _meet_or_join(ep: Episode, closure: bool, op) -> float:
    """Binary adaptation of component-wise evidence rules.

    Node evidence vector e_i = (max(0, llr_i), max(0, -llr_i)) for (H=1, H=0).
    Within a component the vectors are combined coordinatewise with ``op``
    (min = PACT meet; max = cautious-rule analogue for simple support functions);
    components are summed.
    """
    lo = _prior_lo(ep)
    for comp in _components(ep, closure):
        e = np.array([[max(0.0, l), max(0.0, -l)] for l in (_node_loglr(ep, i) for i in comp)])
        agg = op(e, axis=0)
        lo += agg[0] - agg[1]
    return float(sigmoid(lo))


def pact_meet_declared(ep: Episode) -> float:
    return _meet_or_join(ep, closure=False, op=np.min)


def pact_meet_closure(ep: Episode) -> float:
    return _meet_or_join(ep, closure=True, op=np.min)


def cautious_closure(ep: Episode) -> float:
    return _meet_or_join(ep, closure=True, op=np.max)


BASELINES = {
    "naive_bayes": naive_bayes,
    "declared_only": declared_only,
    "pact_meet_declared": pact_meet_declared,
    "pact_meet_closure": pact_meet_closure,
    "cautious_closure": cautious_closure,
    "oracle": oracle,
}
