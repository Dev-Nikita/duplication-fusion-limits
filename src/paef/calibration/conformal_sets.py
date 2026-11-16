"""Calibrated lineage candidate sets (see docs/theory/G2_coverage.md).

Ground truth is read only by the `*_scores` functions, which are applied to CALIBRATION episodes.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from paef.calibration.scorer import LineageScorer, node_probs
from paef.model.lineage import Episode


def conformal_quantile(scores: Sequence[float], delta: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - delta))
    return 1.0 if k > n else float(s[k - 1])


def episode_scores(scorer: LineageScorer, eps: Sequence[Episode], typed: bool) -> List[float]:
    """S(e) = max_j (1 - p̂(true_j)) — joint nonconformity (calibration episodes only)."""
    out = []
    for ep in eps:
        P = node_probs(scorer, ep, typed)
        s = 0.0
        for j, p in P.items():
            codes = ep.meta[j][0]
            s = max(s, 1.0 - p[codes.index(ep.obs[j].true_code)])
        out.append(s)
    return out


def node_scores(scorer: LineageScorer, eps: Sequence[Episode], typed: bool) -> Tuple[List[float], int]:
    """Pooled per-observation scores and max |J| (calibration episodes only)."""
    out, mJ = [], 1
    for ep in eps:
        P = node_probs(scorer, ep, typed)
        mJ = max(mJ, len(P))
        for j, p in P.items():
            codes = ep.meta[j][0]
            out.append(1.0 - p[codes.index(ep.obs[j].true_code)])
    return out, mJ


def build_sets(scorer: LineageScorer, ep: Episode, q: float, typed: bool) -> Dict[int, Tuple[int, ...]]:
    """Ĉ_j = {c : 1 - p̂(c) <= q}; never empty (argmax kept; adding elements cannot hurt coverage)."""
    sets = {}
    for j, p in node_probs(scorer, ep, typed).items():
        codes = ep.meta[j][0]
        keep = [c for c, pc in zip(codes, p) if 1.0 - pc <= q + 1e-12]
        if not keep:
            keep = [codes[int(np.argmax(p))]]
        sets[j] = tuple(keep)
    return sets


def covers(ep: Episode, sets: Dict[int, Tuple[int, ...]]) -> bool:
    return all(ep.obs[j].true_code in s for j, s in sets.items())
