"""Evaluation metrics (probabilistic, selective and duplication sensitivity)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def brier(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return float(np.mean((p - y) ** 2))


def nll(p, y, eps=1e-6):
    p = np.clip(np.asarray(p), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def auroc(p, y):
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def risk_at_coverage(score, pred, y, coverage):
    """Error rate on the `coverage` fraction of cases with the highest confidence score."""
    score, pred, y = map(np.asarray, (score, pred, y))
    n = max(1, int(round(coverage * len(y))))
    order = np.argsort(-score, kind="stable")[:n]
    return float(np.mean(pred[order] != y[order]))


def aurc(score, pred, y):
    score, pred, y = map(np.asarray, (score, pred, y))
    order = np.argsort(-score, kind="stable")
    err = (pred[order] != y[order]).astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def bootstrap_ci(values, n_boot=2000, seed=0, stat=np.mean):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    bs = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
