"""Origin/reading fusion with a variable number of readings per origin (R4 testbed).
Same semantics as paef.real.origami: reading 0 is the declared primary; other readings and copies may
have hidden lineage; copies are re-emissions of firing alarms."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-3


def clip(p):
    return np.clip(p, EPS, 1 - EPS)


@dataclass
class OriginModel:
    theta: np.ndarray   # (2,)
    c: np.ndarray       # (k, 2)

    def marg(self, k):
        return self.theta * self.c[k, 1] + (1 - self.theta) * self.c[k, 0]

    def group_loglik(self, ks, z):
        out = np.zeros(2)
        for h in (0, 1):
            l1, l0 = self.theta[h], 1 - self.theta[h]
            for k in ks:
                l1 *= self.c[k, 1] if z[k] else 1 - self.c[k, 1]
                l0 *= self.c[k, 0] if z[k] else 1 - self.c[k, 0]
            out[h] = np.log(l1 + l0)
        return out


def fit_origin(Z, H, iters=200):
    k = Z.shape[1]
    th = np.array([0.2, 0.7]); c = np.tile([[0.1, 0.8]], (k, 1)).astype(float)
    for _ in range(iters):
        lu1 = np.log(clip(th[H])); lu0 = np.log(clip(1 - th[H]))
        for j in range(k):
            lu1 = lu1 + np.where(Z[:, j] == 1, np.log(c[j, 1]), np.log(1 - c[j, 1]))
            lu0 = lu0 + np.where(Z[:, j] == 1, np.log(c[j, 0]), np.log(1 - c[j, 0]))
        q = 1 / (1 + np.exp(lu0 - lu1))
        th = clip(np.array([q[H == 0].mean(), q[H == 1].mean()]))
        for j in range(k):
            c[j, 1] = clip((q * Z[:, j]).sum() / max(q.sum(), 1e-9))
            c[j, 0] = clip(((1 - q) * Z[:, j]).sum() / max((1 - q).sum(), 1e-9))
    if c[0, 1] < c[0, 0]:
        c = c[:, ::-1].copy(); th = 1 - th
    return OriginModel(th, c)


@dataclass
class Window:
    y: int
    z: Dict[str, np.ndarray]
    hidden_read: Dict[str, List[int]] = field(default_factory=dict)
    copies: List[Tuple[str, int, bool]] = field(default_factory=list)


def make_window(z, y, D, h, rng):
    w = Window(y=y, z=z)
    for o, zz in z.items():
        w.hidden_read[o] = [k for k in range(1, len(zz)) if rng.random() < h]
        for k in range(len(zz)):
            if zz[k] == 1:
                for _ in range(D):
                    w.copies.append((o, k, bool(rng.random() < h)))
    return w


class Layout:
    def __init__(self, origins: Dict[str, List[str]]):
        self.origins = origins
        self.slots = [(o, k) for o, rs in origins.items() for k in range(len(rs))]
        self.index = {s: i for i, s in enumerate(self.slots)}

    def count_features(self, w):          # lineage-blind: every report counted
        f = np.zeros((len(self.slots), 2))
        for o, zz in w.z.items():
            for k, v in enumerate(zz):
                f[self.index[(o, k)], v] += 1
        for o, k, _ in w.copies:
            f[self.index[(o, k)], 1] += 1
        return f.ravel()

    def unit_features(self, w, hidden_copies_as="exclude"):
        f = np.zeros((len(self.slots), 2))
        for o, zz in w.z.items():
            for k, v in enumerate(zz):
                f[self.index[(o, k)], v] += 1
        add = {"exclude": 0.0, "include": 1.0, "half": 0.5}[hidden_copies_as]
        for o, k, hid in w.copies:
            if hid:
                f[self.index[(o, k)], 1] += add
        return f.ravel()

    def lineage_stacking(self, w, lr):
        coef = lr.coef_.ravel()
        base = float(lr.intercept_[0] + self.unit_features(w) @ coef)
        lo = hi = base
        for o, k, hid in w.copies:
            if hid:
                j = self.index[(o, k)] * 2 + 1
                lo += min(0.0, coef[j]); hi += max(0.0, coef[j])
        sig = lambda x: float(1 / (1 + np.exp(-x)))
        mid = float(lr.intercept_[0] + self.unit_features(w, "half") @ coef)
        return sig(mid), sig(lo), sig(hi)


def fuse_generative(w, models, prior):
    b = np.log(prior / (1 - prior)); oracle = naive = bma = 0.0
    for o, zz in w.z.items():
        m = models[o]; k = len(zz)
        g = m.group_loglik(list(range(k)), zz); oracle += g[1] - g[0]
        llr = [np.log(m.marg(j)[1] / m.marg(j)[0]) if zz[j] else np.log((1 - m.marg(j)[1]) / (1 - m.marg(j)[0])) for j in range(k)]
        naive += sum(llr)
        hid = w.hidden_read.get(o, []); decl = [j for j in range(k) if j not in hid]
        l1s, l0s = [], []
        for ch in itertools.product((0, 1), repeat=len(hid)):
            on = decl + [j for j, c in zip(hid, ch) if c]; off = [j for j, c in zip(hid, ch) if not c]
            g = m.group_loglik(on, zz)
            f1 = sum(np.log(m.marg(j)[1] if zz[j] else 1 - m.marg(j)[1]) for j in off)
            f0 = sum(np.log(m.marg(j)[0] if zz[j] else 1 - m.marg(j)[0]) for j in off)
            l1s.append(g[1] + f1); l0s.append(g[0] + f0)
        bma += np.logaddexp.reduce(l1s) - np.logaddexp.reduce(l0s)
    for o, k, hid in w.copies:
        m = models[o]; p = m.marg(k); l = np.log(p[1] / p[0]); naive += l
        if hid:
            bma += np.log(0.5 + 0.5 * p[1]) - np.log(0.5 + 0.5 * p[0])
    sig = lambda x: float(1 / (1 + np.exp(-x)))
    return {"oracle": sig(b + oracle), "naive_bayes": sig(b + naive), "paef_bma": sig(b + bma)}


# ------------------------------------------------------------------ baselines (frozen before R5 data)
def _llr(m, k, z):
    p = m.marg(k)
    return float(np.log(p[1] / p[0])) if z else float(np.log((1 - p[1]) / (1 - p[0])))


def pact_meet(w, models, prior, partition="declared"):
    """PACT operator (Jin et al. 2026), re-implemented from the paper (no public code): evidence vectors
    e = (max(0, llr), max(0, -llr)) in commensurate log-LR units (their Assumption 1); coordinatewise
    meet inside a provenance component, components add (Assumption 2).
    partition='declared': only recorded lineage links observations (hidden readings and hidden copies
    are singleton components — PACT's 'unrecorded shared provenance' case);
    partition='true': the true partition (each origin with all its readings and copies) — reference."""
    ev = lambda l: np.array([max(0.0, l), max(0.0, -l)])
    comps = {}
    singles = []
    for o, zz in w.z.items():
        hid = set(w.hidden_read.get(o, [])) if partition == "declared" else set()
        for k in range(len(zz)):
            e = ev(_llr(models[o], k, zz[k]))
            (singles.append(e) if k in hid else comps.setdefault(o, []).append(e))
    for o, k, hidden in w.copies:
        e = ev(_llr(models[o], k, 1))
        if hidden and partition == "declared":
            singles.append(e)
        else:
            comps.setdefault(o, []).append(e)
    tot = np.zeros(2)
    for es in comps.values():
        tot += np.min(np.array(es), axis=0)
    for e in singles:
        tot += e
    lo = np.log(prior / (1 - prior)) + tot[0] - tot[1]
    return float(1 / (1 + np.exp(-lo)))


def loglinear_pool(w, models, prior, alpha):
    """Log-linear (geometric) pooling of all reports: log-odds = prior + alpha * sum(llr).
    alpha = 1 is naive Bayes; alpha = 1/m is the geometric mean (Chernoff / covariance-intersection type).
    A member of the conservative family discussed by Taylor & Bishop (Information Fusion 2019), adapted to
    binary alarms; alpha < 0 means 'use 1/m'."""
    llrs = []
    for o, zz in w.z.items():
        llrs += [_llr(models[o], k, zz[k]) for k in range(len(zz))]
    llrs += [_llr(models[o], k, 1) for o, k, _ in w.copies]
    a = (1.0 / max(1, len(llrs))) if alpha < 0 else alpha
    lo = np.log(prior / (1 - prior)) + a * sum(llrs)
    return float(1 / (1 + np.exp(-lo)))
