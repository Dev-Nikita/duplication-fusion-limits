"""R2: typed-lineage fusion on ORIGAMI windows (real signalling, constructed observation layer).

Fast exact computation exploits the typed candidate structure:
  * hidden reading k of origin r has candidates {r, FRESH}: enumerate 2^{h_r} choices per origin;
  * hidden copy has candidates {COPY(parent), FRESH}: COPY contributes likelihood 1, FRESH contributes
    the alarm type's marginal; copies never interact with other choices, so they are additive.
Hence min/max/model-average cost O(sum_r 2^{h_r} + #copies) instead of 2^{#hidden}.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

NODES = ["amf", "ausf", "udm", "udr", "smf", "upf", "pcf", "bsf", "gnB"]
STATS = ["p90", "mean", "max"]          # readings; p90 = declared primary
EPS = 1e-3


def clip(p):
    return np.clip(p, EPS, 1 - EPS)


# ------------------------------------------------------------------ latent-class model per origin
@dataclass
class OriginModel:
    theta: np.ndarray   # (2,) P(U=1 | H=h)
    c: np.ndarray       # (3, 2) P(Z_k=1 | U=u)

    def marg(self, k: int) -> np.ndarray:
        """P(Z_k=1 | H=h) for h = 0, 1."""
        return self.theta * self.c[k, 1] + (1 - self.theta) * self.c[k, 0]

    def group_loglik(self, ks: List[int], z: np.ndarray) -> np.ndarray:
        """log P(z_ks | H=h), h=0,1, readings ks conditionally independent given U."""
        out = np.zeros(2)
        for h in (0, 1):
            l1, l0 = self.theta[h], 1 - self.theta[h]
            for k in ks:
                l1 *= self.c[k, 1] if z[k] else 1 - self.c[k, 1]
                l0 *= self.c[k, 0] if z[k] else 1 - self.c[k, 0]
            out[h] = np.log(l1 + l0)
        return out


def fit_origin(Z: np.ndarray, H: np.ndarray, iters: int = 200) -> OriginModel:
    """EM for binary latent U with three binary readings; H observed. Z: (n, 3) in {0,1}."""
    th = np.array([0.2, 0.7])
    c = np.array([[0.1, 0.8]] * 3)
    for _ in range(iters):
        lu1 = np.log(clip(th[H]))
        lu0 = np.log(clip(1 - th[H]))
        for k in range(3):
            lu1 = lu1 + np.where(Z[:, k] == 1, np.log(c[k, 1]), np.log(1 - c[k, 1]))
            lu0 = lu0 + np.where(Z[:, k] == 1, np.log(c[k, 0]), np.log(1 - c[k, 0]))
        q = 1 / (1 + np.exp(lu0 - lu1))  # P(U=1 | z, h)
        th = clip(np.array([q[H == 0].mean(), q[H == 1].mean()]))
        for k in range(3):
            c[k, 1] = clip((q * Z[:, k]).sum() / q.sum())
            c[k, 0] = clip(((1 - q) * Z[:, k]).sum() / (1 - q).sum())
    if c[0, 1] < c[0, 0]:  # label switching: U=1 must be the 'slow' state for the primary reading
        c = c[:, ::-1].copy()
        th = 1 - th
    return OriginModel(theta=th, c=c)


# ------------------------------------------------------------------ alarms and episodes
@dataclass
class Window:
    y: int
    z: Dict[str, np.ndarray]                      # node -> (3,) alarm values
    hidden_read: Dict[str, List[int]] = field(default_factory=dict)   # node -> hidden reading idx (1,2)
    copies: List[Tuple[str, int, bool]] = field(default_factory=list)  # (node, k, hidden)


def thresholds(train_df, rate) -> Dict[Tuple[str, str], float]:
    h = train_df[(train_df.rate == rate) & (train_df.cores == 8)]
    return {(n, s): float(np.nanpercentile(h[f"{n}_{s}"], 90)) for n in NODES for s in STATS}


def alarms(row, thr) -> Dict[str, np.ndarray]:
    out = {}
    for n in NODES:
        v = [row.get(f"{n}_{s}") for s in STATS]
        if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in v):
            continue
        out[n] = np.array([int(v[i] > thr[(n, s)]) for i, s in enumerate(STATS)])
    return out


def make_window(row, thr, y, D, h, rng) -> Window:
    z = alarms(row, thr)
    w = Window(y=y, z=z)
    for n, zz in z.items():
        w.hidden_read[n] = [k for k in (1, 2) if rng.random() < h]
        for k in range(3):
            if zz[k] == 1:
                for _ in range(D):
                    w.copies.append((n, k, bool(rng.random() < h)))
    return w


# ------------------------------------------------------------------ fusion rules
def _llr_marg(m: OriginModel, k: int, z: int) -> float:
    p = m.marg(k)
    return float(np.log(p[1] / p[0])) if z else float(np.log((1 - p[1]) / (1 - p[0])))


def fuse(w: Window, models: Dict[str, OriginModel], prior: float = 0.5) -> Dict[str, float]:
    b = np.log(prior / (1 - prior))
    oracle = declared = naive = geo_sum = 0.0
    n_obs = 0
    lo = hi = bma = 0.0
    pact_decl = pact_clos = 0.0
    for n, zz in w.z.items():
        m = models[n]
        llr = [_llr_marg(m, k, zz[k]) for k in range(3)]
        # oracle: all three readings on origin n
        g = m.group_loglik([0, 1, 2], zz)
        oracle += g[1] - g[0]
        hid = w.hidden_read.get(n, [])
        decl = [k for k in range(3) if k not in hid]
        g = m.group_loglik(decl, zz)
        declared += g[1] - g[0] + sum(llr[k] for k in hid)
        # enumerate hidden readings: each either on origin n or FRESH
        vals, l1s, l0s = [], [], []
        for choice in itertools.product((0, 1), repeat=len(hid)):  # 1 = on origin
            on = decl + [k for k, c in zip(hid, choice) if c]
            off = [k for k, c in zip(hid, choice) if not c]
            g = m.group_loglik(on, zz)
            f1 = sum(np.log(m.marg(k)[1] if zz[k] else 1 - m.marg(k)[1]) for k in off)
            f0 = sum(np.log(m.marg(k)[0] if zz[k] else 1 - m.marg(k)[0]) for k in off)
            l1s.append(g[1] + f1); l0s.append(g[0] + f0); vals.append(g[1] + f1 - g[0] - f0)
        lo += min(vals); hi += max(vals)
        l1s, l0s = np.array(l1s), np.array(l0s)
        bma += (np.logaddexp.reduce(l1s) - np.logaddexp.reduce(l0s))
        naive += sum(llr)
        geo_sum += sum(llr); n_obs += 3
        # PACT-style meet on binary evidence vectors
        ev = lambda l: np.array([max(0, l), max(0, -l)])
        comp_decl = [ev(llr[k]) for k in decl]
        pact_decl += float(np.subtract(*np.min(comp_decl, axis=0)))
        for k in hid:
            pact_decl += float(np.subtract(*ev(llr[k])))
        pact_clos += float(np.subtract(*np.min([ev(l) for l in llr], axis=0)))
    for (n, k, hidden) in w.copies:
        m = models[n]
        l = _llr_marg(m, k, 1)  # copies are only of firing alarms
        naive += l; geo_sum += l; n_obs += 1
        if hidden:
            declared += l
            lo += min(0.0, l); hi += max(0.0, l)
            p = m.marg(k)
            bma += np.log(0.5 + 0.5 * p[1]) - np.log(0.5 + 0.5 * p[0])
            pact_decl += l  # unresolved copy is a separate component under the declared partition
        # declared copies / closure: inside the parent's component, meet unchanged
    sig = lambda x: float(1 / (1 + np.exp(-x)))
    return {"oracle": sig(b + oracle), "declared_only": sig(b + declared), "naive_bayes": sig(b + naive),
            "geo_pool": sig(b + geo_sum / max(1, n_obs)), "paef_bma": sig(b + bma),
            "paef_lower": sig(b + lo), "paef_upper": sig(b + hi),
            "pact_meet_declared": sig(b + pact_decl), "pact_meet_closure": sig(b + pact_clos)}


def count_features(w: Window) -> np.ndarray:
    """Lineage-blind features for stacking: per (node, stat) number of reports with value 1 and 0."""
    f = np.zeros((len(NODES), 3, 2))
    for i, n in enumerate(NODES):
        if n in w.z:
            for k in range(3):
                f[i, k, w.z[n][k]] += 1
    for (n, k, _) in w.copies:
        f[NODES.index(n), k, 1] += 1
    return f.ravel()


# ------------------------------------------------------------------ A4 (exploratory): lineage-aware stacking
def unit_features(w: Window, hidden_copies_as: str = "exclude") -> np.ndarray:
    """Counts over *countable units*: known copies are never counted; hidden copies are excluded
    ('exclude'), counted ('include') or half-counted ('half', uniform prior over {COPY, FRESH})."""
    f = np.zeros((len(NODES), 3, 2))
    for i, n in enumerate(NODES):
        if n in w.z:
            for k in range(3):
                f[i, k, w.z[n][k]] += 1
    add = {"exclude": 0.0, "include": 1.0, "half": 0.5}[hidden_copies_as]
    for (n, k, hidden) in w.copies:
        if hidden:
            f[NODES.index(n), k, 1] += add
    return f.ravel()


def lineage_stacking(w: Window, lr) -> Dict[str, float]:
    """Point (uniform prior over hidden-copy lineage) and certified interval for a linear-logit model.
    The logit is linear in counts, so each hidden copy moves it by 0 (COPY) or w_j (FRESH): the
    extrema over all lineage assignments are additive."""
    coef = lr.coef_.ravel()
    base = float(lr.intercept_[0] + unit_features(w, "exclude") @ coef)
    lo = hi = base
    for (n, k, hidden) in w.copies:
        if hidden:
            j = (NODES.index(n) * 3 + k) * 2 + 1
            lo += min(0.0, coef[j]); hi += max(0.0, coef[j])
    sig = lambda x: float(1 / (1 + np.exp(-x)))
    mid = float(lr.intercept_[0] + unit_features(w, "half") @ coef)
    return {"ls_point": sig(mid), "ls_lower": sig(lo), "ls_upper": sig(hi)}
