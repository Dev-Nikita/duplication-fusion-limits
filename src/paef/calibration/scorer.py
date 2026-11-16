"""Conditional-logit lineage scorer: p(c | X, j) ∝ exp(w · f(j, c)) over j's typed candidates.

Fitted on a training split whose lineage is revealed. Reads `true_code` ONLY in `fit` (training labels).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from paef.model.lineage import Episode, copy_parent, is_copy


class LineageScorer:
    def __init__(self, l2: float = 1e-2):
        self.l2 = l2
        self.w: Optional[np.ndarray] = None

    def fit(self, episodes: Sequence[Episode]) -> "LineageScorer":
        Fs, ys = [], []
        for ep in episodes:
            for j, (codes, F) in ep.meta.items():
                if ep.obs[j].value is None:
                    continue
                Fs.append(F)
                ys.append(codes.index(ep.obs[j].true_code))
        d = Fs[0].shape[1]
        K = max(F.shape[0] for F in Fs)
        X = np.zeros((len(Fs), K, d))
        M = np.zeros((len(Fs), K), dtype=bool)
        for n, F in enumerate(Fs):
            X[n, :F.shape[0]] = F
            M[n, :F.shape[0]] = True
        y = np.asarray(ys)
        idx = np.arange(len(y))

        def nll(w):
            u = np.where(M, X @ w, -np.inf)
            m = u.max(axis=1, keepdims=True)
            e = np.exp(u - m)
            Z = e.sum(axis=1, keepdims=True)
            p = e / Z
            ll = (u[idx, y] - m[:, 0] - np.log(Z[:, 0])).sum()
            g = (X[idx, y] - np.einsum("nk,nkd->nd", p, X)).sum(axis=0)
            return -ll + self.l2 * w @ w, -g + 2 * self.l2 * w

        r = minimize(nll, np.zeros(d), jac=True, method="L-BFGS-B")
        self.w = r.x
        return self

    def probs(self, F: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        u = F @ self.w
        if mask is not None:
            u = np.where(mask, u, -np.inf)
        u = u - u[np.isfinite(u)].max()
        p = np.exp(u)
        return p / p.sum()


def consistency_mask(ep: Episode, j: int, codes: Sequence[int]) -> np.ndarray:
    """Value-consistent candidates: COPY(p) requires z_j == z_p (when z_p observed)."""
    zj = ep.obs[j].value
    m = np.ones(len(codes), dtype=bool)
    for k, c in enumerate(codes):
        if is_copy(c):
            zp = ep.obs[copy_parent(c)].value
            if zp is not None and zj is not None and zp != zj:
                m[k] = False
    return m


def node_probs(scorer: LineageScorer, ep: Episode, typed: bool) -> Dict[int, np.ndarray]:
    out = {}
    for j, (codes, F) in ep.meta.items():
        if ep.obs[j].value is None:
            continue
        out[j] = scorer.probs(F, consistency_mask(ep, j, codes) if typed else None)
    return out
