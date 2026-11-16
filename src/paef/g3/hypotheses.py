"""Hypothesis table for lineage queries: every admissible assignment of the unresolved observations,
with its certified log-odds range (lo, hi) and prior weight. No ground truth is read here."""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from paef.fusion.bounds import clusters
from paef.model.lineage import FRESH, Episode, loglik_pair


@dataclass
class HypTable:
    nodes: List[int]            # unresolved observed indices
    codes: np.ndarray           # (H, |nodes|) assignment codes
    lo: np.ndarray              # (H,) log-odds lower (FRESH box vertex)
    hi: np.ndarray              # (H,) log-odds upper
    w: np.ndarray               # (H,) prior weights (normalised)
    tau_lo: float               # threshold in log-odds
    single_per_cluster: bool    # Prop. G3.1 regime


def _vertex(e: Episode, i: int, side: str):
    o = e.obs[i]
    if o.fresh_box is None:
        return o.fresh_theta
    (lo0, hi0), (lo1, hi1) = o.fresh_box
    up = (o.value == 1) == (side == "hi")
    return (lo0, hi1) if up else (hi0, lo1)


def build_table(e: Episode, sets: Dict[int, Tuple[int, ...]], probs: Dict[int, Dict[int, float]],
                tau: float = 0.5) -> HypTable:
    """`e` must already carry leakage-free parameters (make_agnostic(..., fresh_box=True))."""
    nodes = [j for j in sorted(sets) if e.obs[j].value is not None]
    fixed = {i: e.obs[i].candidates[0] for i in e.observed() if i not in sets}
    prior_lo = float(np.log(e.prior) - np.log1p(-e.prior))
    rows, los, his, ws = [], [], [], []
    for combo in itertools.product(*[sets[j] for j in nodes]):
        assign = dict(fixed)
        assign.update(zip(nodes, combo))
        out = []
        for side in ("lo", "hi"):
            saved = {}
            for i, c in assign.items():
                if c == FRESH:
                    saved[i] = e.obs[i].fresh_theta
                    e.obs[i].fresh_theta = _vertex(e, i, side)
            r = loglik_pair(e, assign)
            for i, th in saved.items():
                e.obs[i].fresh_theta = th
            out.append(None if r is None else prior_lo + r[0] - r[1])
        if out[0] is None:
            continue
        rows.append(combo); los.append(out[0]); his.append(out[1])
        ws.append(float(np.prod([probs[j].get(c, 0.0) for j, c in zip(nodes, combo)])))
    w = np.asarray(ws)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / max(1, len(w)))
    # Prop. G3.1 regime: each cluster holds at most one unresolved node
    e2 = copy.deepcopy(e)
    for j in nodes:
        e2.obs[j].candidates = sets[j]
    single = all(sum(1 for i in cl if i in nodes) <= 1 for cl in clusters(e2))
    return HypTable(nodes, np.asarray(rows, dtype=int).reshape(len(rows), len(nodes)),
                    np.asarray(los), np.asarray(his), w, float(np.log(tau) - np.log1p(-tau)), single)


def status(t: HypTable, mask: np.ndarray) -> int:
    """1 / 0 robust decision, -1 undecided, on the hypotheses kept by mask."""
    if not mask.any():
        return -1
    if t.lo[mask].min() > t.tau_lo:
        return 1
    if t.hi[mask].max() < t.tau_lo:
        return 0
    return -1
