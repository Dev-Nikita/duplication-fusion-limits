"""Per-entity unit layer (R5b): origins are UEs, every UE has the same reading types, and slots are
reading TYPES shared by all UEs. Counts per slot therefore range over the number of affected UEs, which
is informative; copies are derived re-emissions. Frozen before R5 metrics were computed."""
from __future__ import annotations
import numpy as np


class UnitLayout:
    def __init__(self, n_readings: int):
        self.K = n_readings

    def count_features(self, w):          # lineage-blind: every message counted
        f = np.zeros((self.K, 2))
        for zz in w.z.values():
            for k, v in enumerate(zz):
                f[k, v] += 1
        for _, k, _ in w.copies:
            f[k, 1] += 1
        return f.ravel()

    def presence_features(self, w):       # lineage-free normalisation: is the alarm type present at all
        f = self.count_features(w).reshape(self.K, 2)
        return (f[:, 1] > 0).astype(float)

    def log_features(self, w):
        return np.log1p(self.count_features(w))

    def unit_features(self, w, hidden_copies_as="exclude"):
        f = np.zeros((self.K, 2))
        for zz in w.z.values():
            for k, v in enumerate(zz):
                f[k, v] += 1
        add = {"exclude": 0.0, "include": 1.0, "half": 0.5}[hidden_copies_as]
        for _, k, hid in w.copies:
            if hid:
                f[k, 1] += add
        return f.ravel()

    def lineage_stacking(self, w, lr):
        coef = lr.coef_.ravel()
        base = float(lr.intercept_[0] + self.unit_features(w) @ coef)
        lo = hi = base
        for _, k, hid in w.copies:
            if hid:
                j = 2 * k + 1
                lo += min(0.0, coef[j]); hi += max(0.0, coef[j])
        sig = lambda x: float(1 / (1 + np.exp(-x)))
        mid = float(lr.intercept_[0] + self.unit_features(w, "half") @ coef)
        return sig(mid), sig(lo), sig(hi)
