"""Lineage (provenance) model for diagnostic evidence.

Generative structure (binary incident hypothesis H):

    H  ->  U_r  (latent origin signal of root r)  ->  Z_i  (observation i)

* P(U_r = 1 | H = h) = theta[r][h]
* P(Z_i = 1 | U = u) = chan[i][u]   (channel of observation i; an exact copy has chan = (0, 1))
* Observations that share a root are conditionally independent given U_r,
  NOT given H. Observations on different roots are conditionally independent given H.

Provenance knowledge is incomplete: each observation i carries a candidate set C_i of
roots it may derive from. A singleton C_i means known provenance. A candidate
``FRESH`` means "i may be an independent origin of its own"; in that case the
observation behaves as a root with parameters ``fresh_theta[i]`` and its own channel.

An *assignment* rho picks one element of C_i for every i. The admissible set of
generative models is {model(rho) : rho in prod_i C_i}. Bounds on P(H=1 | z) are the
min / max of the posterior over admissible assignments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FRESH = -1  # sentinel: observation is its own independent origin


def COPY(i: int) -> int:
    """Candidate code: 'this observation is an information-free copy of observation i'."""
    return -2 - i


def is_copy(code: int) -> bool:
    return code <= -2


def copy_parent(code: int) -> int:
    return -2 - code


@dataclass
class Observation:
    value: Optional[int]                 # 0 / 1, or None if missing
    chan: Tuple[float, float]            # (P(Z=1|U=0), P(Z=1|U=1))
    candidates: Tuple[int, ...]          # candidate roots; may contain FRESH
    fresh_theta: Tuple[float, float] = (0.1, 0.9)  # (P(U=1|H=0), P(U=1|H=1)) if FRESH
    true_root: Optional[int] = None      # ground truth origin root, never read by fusion code
    true_code: Optional[int] = None      # ground truth candidate code (root or COPY(i))
    kind: str = "derived"                # exact | derived | independent  (bookkeeping only)
    fresh_box: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None  # ((lo0,hi0),(lo1,hi1)) credal box for FRESH origin
    calib: Optional[Tuple[float, float]] = None  # deployable marginal (P(Z=1|H=0), P(Z=1|H=1)); if set, used instead of truth


@dataclass
class Episode:
    prior: float                         # P(H = 1)
    theta: Dict[int, Tuple[float, float]]  # root -> (P(U=1|H=0), P(U=1|H=1))
    obs: List[Observation] = field(default_factory=list)
    label: Optional[int] = None          # ground truth H, never read by fusion code

    # ------------------------------------------------------------------ helpers
    def observed(self) -> List[int]:
        return [i for i, o in enumerate(self.obs) if o.value is not None]

    def n_assignments(self) -> int:
        n = 1
        for i in self.observed():
            n *= len(self.obs[i].candidates)
        return n


def _root_lik(theta_h: float, members: Sequence[Observation]) -> float:
    """P(z_members | H=h) for one root with P(U=1|H=h)=theta_h."""
    l1 = theta_h
    l0 = 1.0 - theta_h
    for o in members:
        c0, c1 = o.chan
        if o.value == 1:
            l1 *= c1
            l0 *= c0
        else:
            l1 *= 1.0 - c1
            l0 *= 1.0 - c0
    return l0 + l1


def group_assignment(ep: Episode, assign: Dict[int, int]):
    """Turn an assignment into likelihood groups.

    Returns a list of (theta0, theta1, members) or None if the assignment is
    inadmissible (a copy disagrees with the value of its declared parent).
    Copies of observed parents carry no information and are dropped; a copy of a
    missing parent stands in for that parent.
    """
    # resolve copy chains to an effective origin code
    eff: Dict[int, Tuple[int, int]] = {}  # i -> (code, carrier index whose fresh_theta applies)

    def resolve(i, depth=0):
        if i in eff:
            return eff[i]
        code = assign[i] if i in assign else ep.obs[i].candidates[0]
        if is_copy(code) and depth < 50:
            p = copy_parent(code)
            if ep.obs[p].value is not None and p in assign:
                eff[i] = ("dup", p)
            else:
                pc = resolve(p, depth + 1) if p in assign or len(ep.obs[p].candidates) == 1 else (FRESH, p)
                eff[i] = pc
            return eff[i]
        eff[i] = (code, i)
        return eff[i]

    groups: Dict[object, list] = {}
    for i in assign:
        code, carrier = resolve(i)
        if code == "dup":
            if ep.obs[i].value != ep.obs[carrier].value:
                return None
            continue
        key = ("f", carrier) if code == FRESH else ("r", code)
        groups.setdefault(key, []).append(ep.obs[i])
    out = []
    for key, members in groups.items():
        th0, th1 = ep.obs[key[1]].fresh_theta if key[0] == "f" else ep.theta[key[1]]
        out.append((th0, th1, members))
    return out


def loglik_pair(ep: Episode, assign: Dict[int, int]):
    """(log P(z|H=1), log P(z|H=0)) under an assignment, or None if inadmissible."""
    g = group_assignment(ep, assign)
    if g is None:
        return None
    a1 = a0 = 0.0
    for th0, th1, members in g:
        a1 += np.log(max(_root_lik(th1, members), 1e-300))
        a0 += np.log(max(_root_lik(th0, members), 1e-300))
    return a1, a0


def log_odds_given_assignment(ep: Episode, assign: Dict[int, int]) -> float:
    """Exact posterior log-odds under a full assignment (observed index -> code)."""
    a1, a0 = loglik_pair(ep, assign)
    return float(np.log(ep.prior) - np.log1p(-ep.prior) + a1 - a0)


def marginal_lik(ep: Episode, i: int) -> Tuple[float, float]:
    """Calibrated marginal P(Z_i=1|H=0), P(Z_i=1|H=1) for observation i on its TRUE origin.

    This is what a per-source calibration would estimate; naive fusion uses it.
    """
    o = ep.obs[i]
    if o.calib is not None:
        return o.calib
    if o.true_root is None or o.true_root < 0:
        th0, th1 = o.fresh_theta
    else:
        th0, th1 = ep.theta[o.true_root]
    c0, c1 = o.chan
    m0 = th0 * c1 + (1 - th0) * c0
    m1 = th1 * c1 + (1 - th1) * c0
    return m0, m1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
