"""Pre-described computational sensitivity analysis (docs/sim-protocol.md). Supplementary: it changes none of
the R4-R11-X conclusions. A generative model independent of the testbed; nothing is tuned toward a result.

Window: M reporting units (population known: M = M0; unknown: M ~ U{M0/2..M0}). Class y ~ Bernoulli(0.5).
If y = 1, a random fraction rho of units is affected and fires with prob 0.8; all other units fire with
background prob 0.1. Every firing unit sends one firing message. Every non-firing unit sends one negative
report, which is lost with prob q. Silencing (axis 'sil'): with prob sil each affected unit sends nothing at all
(neither firing message nor negative report), as when the fault stops the reporter.
Duplication of firing messages:
  LI training: D ~ {0,1,2,4,8} copies per firing message (i.i.d. multiplicity augmentation, as in the testbed)
  LD test pair: a window-level storm with prob 0.5 + kappa/2 if y = 1 else 0.5 - kappa/2 ('inc'), and the
  mirrored pair ('ben'); a storm adds 16 copies per firing message.
Channels for the lineage-free learner: A (firing messages), F (A + negative reports), A+U (A + distinct firing
units). Reference: U (distinct firing units = exact lineage). Learner: HistGradientBoosting with fixed
hyperparameters (depth 3, 200 iterations, lr 0.1) for every channel; 4000 training and 4000 test windows.
Outcome: Brier(channel) - Brier(U), worst case over the LD pair; 5 replications per cell.
Usage: python3 -I sensitivity.py <out.csv>"""
import itertools, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

M0, N_TR, N_TE, REPS, STORM = 20, 4000, 4000, 5, 16
GRID = dict(rho=(0.1, 0.3, 0.6, 0.9), kappa=(0.0, 0.3, 0.6), q=(0.0, 0.1, 0.3, 0.6), pop=("known", "unknown"),
            sil=(0.0, 0.5))


def sample(n, rho, q, pop, sil, rng, dup):
    """Vectorised over windows (same model as the per-window description in the docstring)."""
    y = rng.integers(0, 2, n)
    M = np.full(n, M0) if pop == "known" else rng.integers(M0 // 2, M0 + 1, n)
    present = np.arange(M0)[None, :] < M[:, None]
    aff = present & (rng.random((n, M0)) < np.where(y[:, None] == 1, rho, 0.0))
    silent = aff & (rng.random((n, M0)) < sil)
    fire = present & np.where(aff, rng.random((n, M0)) < 0.8, rng.random((n, M0)) < 0.1) & ~silent
    neg = present & ~fire & ~silent & (rng.random((n, M0)) >= q)
    u = fire.sum(1)
    X = {"U": u.astype(float), "F_neg": neg.sum(1).astype(float), "A": (u + dup(y, fire, rng)).astype(float)}
    return y, X


def feats(X, ch):
    if ch == "A":
        return X["A"][:, None]
    if ch == "F":
        return np.c_[X["A"], X["F_neg"]]
    if ch == "A+U":
        return np.c_[X["A"], X["U"]]
    return X["U"][:, None]


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def main():
    rows = []
    for cell, (rho, kappa, q, pop, sil) in enumerate(itertools.product(*GRID.values())):
        for rep in range(REPS):
            rng = np.random.default_rng(20261024 + 1000 * cell + rep)
            li = lambda y, fire, r: (r.choice((0, 1, 2, 4, 8), size=fire.shape) * fire).sum(1)  # noqa: E731
            ytr, Xtr = sample(N_TR, rho, q, pop, sil, rng, li)
            models = {ch: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1, random_state=0)
                      .fit(feats(Xtr, ch), ytr) for ch in ("A", "F", "A+U", "U")}
            gaps = {ch: [] for ch in ("A", "F", "A+U")}
            for sign in (+1, -1):
                def ld(y, fire, r, s=sign):
                    p = np.where(y == 1, 0.5 + s * kappa / 2, 0.5 - s * kappa / 2)
                    return STORM * fire.sum(1) * (r.random(len(y)) < p)
                yte, Xte = sample(N_TE, rho, q, pop, sil, rng, ld)
                bu = brier(models["U"].predict_proba(feats(Xte, "U"))[:, 1], yte)
                for ch in gaps:
                    gaps[ch].append(brier(models[ch].predict_proba(feats(Xte, ch))[:, 1], yte) - bu)
            rows.append(dict(rho=rho, kappa=kappa, q=q, pop=pop, sil=sil, rep=rep,
                             **{f"gap_{ch}": max(v) for ch, v in gaps.items()}))
        print(rho, kappa, q, pop, sil, flush=True)
    pd.DataFrame(rows).to_csv(sys.argv[1], index=False)


if __name__ == "__main__":
    main()
