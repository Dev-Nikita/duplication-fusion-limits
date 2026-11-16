"""Provenance-interval fusion: posterior bounds over all admissible lineage assignments.

Exact enumeration (pilot scale). Enumeration factorises over interaction clusters:
observations whose candidate sets can share a root, or that may be copies of one
another, are enumerated jointly; disjoint clusters add in log-odds, so min / max
and the uniform model average decompose cluster by cluster.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from paef.model.lineage import (FRESH, Episode, copy_parent, is_copy, loglik_pair,
                                sigmoid)


def clusters(ep: Episode) -> List[List[int]]:
    idx = ep.observed()
    allidx = list(range(len(ep.obs)))
    parent = {i: i for i in allidx}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    root_anchor: Dict[int, int] = {}
    for i in idx:
        for c in ep.obs[i].candidates:
            if c == FRESH:
                continue
            if is_copy(c):
                union(i, copy_parent(c))
            else:
                if c in root_anchor:
                    union(i, root_anchor[c])
                else:
                    root_anchor[c] = i
    out: Dict[int, List[int]] = {}
    for i in idx:
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _cluster_table(ep: Episode, cl: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    cand = [ep.obs[i].candidates for i in cl]
    l1s, l0s = [], []
    for combo in itertools.product(*cand):
        r = loglik_pair(ep, dict(zip(cl, combo)))
        if r is None:
            continue
        l1s.append(r[0])
        l0s.append(r[1])
    return np.asarray(l1s), np.asarray(l0s)


def _with_fresh_vertex(ep: Episode, cl: List[int], side: str):
    """Set fresh_theta of cluster members to the vertex of their credal box that minimises
    (side='lo') or maximises (side='hi') the single-observation likelihood ratio.
    A FRESH origin always forms a singleton group, and its LR is monotone in each of
    theta0, theta1 (for c1 >= c0), so the extremum over the box is attained at a vertex."""
    saved = {}
    for i in cl:
        o = ep.obs[i]
        if o.fresh_box is None or FRESH not in o.candidates:
            continue
        (lo0, hi0), (lo1, hi1) = o.fresh_box
        up = (o.value == 1) == (side == "hi")  # want large LR for z=1 on hi side, etc.
        saved[i] = o.fresh_theta
        o.fresh_theta = (lo0, hi1) if up else (hi0, lo1)
    return saved


def _restore(ep: Episode, saved):
    for i, th in saved.items():
        ep.obs[i].fresh_theta = th


def _lme(x):
    m = x.max()
    return float(m + np.log(np.mean(np.exp(x - m))))


def posterior_bounds(ep: Episode) -> Dict[str, float]:
    """Lower / upper P(H=1|z) over admissible assignments, plus the uniform average."""
    prior_lo = float(np.log(ep.prior) - np.log1p(-ep.prior))
    lo = hi = bma = prior_lo
    n_assign = 1
    for cl in clusters(ep):
        l1, l0 = _cluster_table(ep, cl)
        if len(l1) == 0:  # every assignment contradicted by data: metadata is wrong
            continue
        v = l1 - l0
        if any(ep.obs[i].fresh_box is not None for i in cl):
            sv = _with_fresh_vertex(ep, cl, "lo")
            a1, a0 = _cluster_table(ep, cl)
            _restore(ep, sv)
            sv = _with_fresh_vertex(ep, cl, "hi")
            b1, b0 = _cluster_table(ep, cl)
            _restore(ep, sv)
            lo += (a1 - a0).min()
            hi += (b1 - b0).max()
        else:
            lo += v.min()
            hi += v.max()
        bma += _lme(l1) - _lme(l0)
        n_assign *= len(v)
    return {"lower": float(sigmoid(lo)), "upper": float(sigmoid(hi)), "bma": float(sigmoid(bma)),
            "lo_logodds": float(lo), "hi_logodds": float(hi), "n_assign": n_assign}


def resolve_value(ep: Episode, i: int) -> float:
    """Expected reduction of the log-odds interval width if i's origin is revealed
    (uniform over i's candidates; no ground truth is read)."""
    base = posterior_bounds(ep)
    w0 = base["hi_logodds"] - base["lo_logodds"]
    o = ep.obs[i]
    if len(o.candidates) <= 1 or o.value is None:
        return 0.0
    orig = o.candidates
    ws = []
    for r in orig:
        o.candidates = (r,)
        b = posterior_bounds(ep)
        ws.append(b["hi_logodds"] - b["lo_logodds"])
    o.candidates = orig
    return float(w0 - np.mean(ws))


def weighted_posterior(ep: Episode, weights) -> float:
    """Model average with per-observation lineage weights w_j(c) (product prior over assignments).

    ``weights``: {j: {code: w}} for unresolved observations; others get uniform weight.
    Assignments with zero weight or zero likelihood are skipped.
    """
    prior_lo = float(np.log(ep.prior) - np.log1p(-ep.prior))
    lo = prior_lo
    for cl in clusters(ep):
        cand = [ep.obs[i].candidates for i in cl]
        l1s, l0s = [], []
        for combo in itertools.product(*cand):
            lw = 0.0
            ok = True
            for i, c in zip(cl, combo):
                if i in weights:
                    w = weights[i].get(c, 0.0)
                    if w <= 0:
                        ok = False
                        break
                    lw += np.log(w)
            if not ok:
                continue
            r = loglik_pair(ep, dict(zip(cl, combo)))
            if r is None:
                continue
            l1s.append(r[0] + lw)
            l0s.append(r[1] + lw)
        if not l1s:
            continue
        l1s, l0s = np.asarray(l1s), np.asarray(l0s)
        lo += float(np.logaddexp.reduce(l1s) - np.logaddexp.reduce(l0s))
    return float(sigmoid(lo))
