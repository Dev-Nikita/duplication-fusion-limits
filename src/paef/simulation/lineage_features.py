"""G2 generator: full typed candidate universe + per-candidate lineage metadata.

Built on `generator.sample_episode` (same generative model for H, U, Z). For every secondary
observation selected as unresolved (prob. `hide`), the candidate set is replaced by the full typed
universe of the episode:
  * a derived reading  -> {every root r} ∪ {FRESH}
  * an exact copy      -> {COPY(primary of every root)} ∪ {FRESH}
The type (copy vs reading) is treated as known (payload identity is observable).

Metadata features per (observation j, candidate c), a vector f(j, c) of length 4:
  [lag, same_procedure, is_fresh, is_fresh * orphan_flag]
  * true non-fresh candidate : lag ~ Exp(mean lag_true), same_procedure ~ Bern(0.75)
  * other non-fresh candidates: lag ~ Exp(mean 2.5),    same_procedure ~ Bern(0.30)
  * FRESH candidate          : lag = 0, same_procedure = 0, is_fresh = 1
  * orphan_flag (observation-level): Bern(0.70) if j truly is an independent origin, else Bern(0.25)
Ground truth stays in `true_code`; features are stored in `ep.meta[j] = (codes, F)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from paef.model.lineage import COPY, FRESH, Episode
from paef.simulation.generator import GenConfig, sample_episode


@dataclass
class FeatConfig:
    gen: GenConfig = field(default_factory=lambda: GenConfig(hide=0.5, p_true_fresh=0.3,
                                                             k_roots=(2, 3), sec_per_root=(1, 2)))
    lag_true: float = 1.0
    lag_other: float = 2.5


def sample_featured(rng: np.random.Generator, cfg: FeatConfig) -> Episode:
    ep = sample_episode(rng, cfg.gen)
    roots = sorted(ep.theta)
    prim = {}
    for i, o in enumerate(ep.obs):
        if o.kind == "primary":
            prim[o.true_root] = i
    ep.meta = {}
    for j, o in enumerate(ep.obs):
        if len(o.candidates) <= 1:
            continue
        if o.kind == "exact":
            codes = [COPY(prim[r]) for r in roots] + [FRESH]
        else:
            codes = [r for r in roots] + [FRESH]
        truly_fresh = o.true_code == FRESH
        orphan = float(rng.random() < (0.70 if truly_fresh else 0.25))
        F = np.zeros((len(codes), 4))
        for k, c in enumerate(codes):
            if c == FRESH:
                F[k] = [0.0, 0.0, 1.0, orphan]
            elif c == o.true_code:
                F[k] = [rng.exponential(cfg.lag_true), float(rng.random() < 0.75), 0.0, 0.0]
            else:
                F[k] = [rng.exponential(cfg.lag_other), float(rng.random() < 0.30), 0.0, 0.0]
        o.candidates = tuple(int(c) for c in codes)
        ep.meta[j] = (tuple(int(c) for c in codes), F)
    return ep
