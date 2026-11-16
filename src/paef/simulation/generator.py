"""Synthetic lineage episodes (pilot generator; independent of every other project).

An episode has K origins (roots). Each root emits one primary observation and a
number of secondary observations of two kinds:
  * exact copies of the primary (re-published / re-forwarded alerts): carry no new
    information about H;
  * derived observations: a separate noisy reading of the same origin signal U_r:
    carry partial new information (they reduce channel noise, not origin noise).
A fraction ``hide`` of secondary observations has unresolved provenance; their
candidate set contains the true origin, ``n_decoys`` decoys of the same kind on other
roots, and FRESH (= "independent origin"). With probability ``misspec`` the true
origin is dropped from the candidate set (wrong provenance metadata).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from paef.model.lineage import COPY, FRESH, Episode, Observation


@dataclass
class GenConfig:
    prior: float = 0.3
    k_roots: tuple = (2, 3)
    sec_per_root: tuple = (1, 3)
    p_exact: float = 0.4
    hide: float = 0.5
    misspec: float = 0.0
    p_missing: float = 0.05
    n_decoys: int = 1
    allow_fresh: bool = True
    p_true_fresh: float = 0.0    # share of hidden secondaries that truly are independent origins


def _theta(rng):
    return (rng.uniform(0.03, 0.30), rng.uniform(0.65, 0.95))


def _chan(rng):
    return (rng.uniform(0.02, 0.25), rng.uniform(0.70, 0.98))


def sample_episode(rng: np.random.Generator, cfg: GenConfig,
                   force_label: Optional[int] = None) -> Episode:
    h = int(rng.random() < cfg.prior) if force_label is None else int(force_label)
    k = int(rng.integers(cfg.k_roots[0], cfg.k_roots[1] + 1))
    theta = {r: _theta(rng) for r in range(k)}
    ep = Episode(prior=cfg.prior, theta=theta, label=h)
    primary_idx = {}
    u = {r: int(rng.random() < theta[r][h]) for r in range(k)}
    # primaries first so copy codes can reference them
    for r in range(k):
        ch = _chan(rng)
        primary_idx[r] = len(ep.obs)
        ep.obs.append(Observation(value=int(rng.random() < ch[u[r]]), chan=ch, candidates=(r,),
                                  fresh_theta=theta[r], true_root=r, true_code=r, kind="primary"))
    for r in range(k):
        n_sec = int(rng.integers(cfg.sec_per_root[0], cfg.sec_per_root[1] + 1))
        for _ in range(n_sec):
            exact = rng.random() < cfg.p_exact
            if exact:
                p = primary_idx[r]
                ch, z, code = ep.obs[p].chan, ep.obs[p].value, COPY(p)
            else:
                ch = _chan(rng)
                z = int(rng.random() < ch[u[r]])
                code = r
                if rng.random() < cfg.p_missing:
                    z = None
            cands = (code,)
            hidden = rng.random() < cfg.hide
            if hidden and cfg.allow_fresh and cfg.p_true_fresh > 0 and rng.random() < cfg.p_true_fresh:
                # truly independent corroboration of the same kind as root r
                u_new = int(rng.random() < theta[r][h])
                if not exact:
                    ch = _chan(rng)
                z = int(rng.random() < ch[u_new])
                code = FRESH
            if hidden:
                others = [q for q in range(k) if q != r]
                nd = min(cfg.n_decoys, len(others))
                decoys = [int(d) for d in rng.choice(others, size=nd, replace=False)] if nd else []
                lookalike = COPY(primary_idx[r]) if exact else r
                pool = ([code] if code != FRESH else [lookalike]) + \
                       [COPY(primary_idx[d]) if exact else d for d in decoys]
                if code != FRESH and rng.random() < cfg.misspec and len(pool) > 1:
                    pool = pool[1:]
                if cfg.allow_fresh:
                    pool.append(FRESH)
                rng.shuffle(pool)
                cands = tuple(int(c) for c in pool)
            ep.obs.append(Observation(value=z, chan=ch, candidates=cands, fresh_theta=theta[r],
                                      true_root=r if code != FRESH else -1, true_code=code,
                                      kind="exact" if exact else "derived"))
    return ep
