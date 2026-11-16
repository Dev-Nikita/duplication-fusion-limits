"""Leakage-free parameterisation (amendment A2).

The pilot generator sets ``fresh_theta`` of every observation to the origin parameters of its
TRUE root, and baselines read per-observation marginals computed from the true root. Both are
derived from ground truth. ``make_agnostic`` replaces them with quantities a deployed system
could compute without knowing the lineage:

* fresh_theta  := mean origin parameters over the episode's declared roots;
* calib (used by naive / PACT / cautious baselines) := for singleton candidates, the marginal
  under the declared origin; for unresolved observations, the average marginal over the
  candidate origins (FRESH contributes the mean-parameter origin; a copy candidate contributes
  its parent's declared marginal).
"""
from __future__ import annotations

import copy

import numpy as np

from paef.model.lineage import FRESH, Episode, copy_parent, is_copy


def _marg(theta, chan):
    th0, th1 = theta
    c0, c1 = chan
    return (th0 * c1 + (1 - th0) * c0, th1 * c1 + (1 - th1) * c0)


def make_agnostic(ep: Episode, fresh_box: bool = False) -> Episode:
    e = copy.deepcopy(ep)
    ths = np.array(list(e.theta.values()))
    mean_th = (float(ths[:, 0].mean()), float(ths[:, 1].mean()))
    box = ((float(ths[:, 0].min()), float(ths[:, 0].max())),
           (float(ths[:, 1].min()), float(ths[:, 1].max())))
    for o in e.obs:
        o.fresh_theta = mean_th
        if fresh_box:
            o.fresh_box = box

    def declared_marg(i, depth=0):
        o = e.obs[i]
        ms = []
        for c in o.candidates:
            if c == FRESH:
                ms.append(_marg(mean_th, o.chan))
            elif is_copy(c):
                ms.append(declared_marg(copy_parent(c), depth + 1) if depth < 20 else _marg(mean_th, o.chan))
            else:
                ms.append(_marg(e.theta[c], o.chan))
        ms = np.array(ms)
        return (float(ms[:, 0].mean()), float(ms[:, 1].mean()))

    for i, o in enumerate(e.obs):
        o.calib = declared_marg(i)
    return e
