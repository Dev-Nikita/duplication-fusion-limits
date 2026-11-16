"""EXPLORATORY (after the frozen sensitivity run): is a fuser given the denominator INSTEAD of the firing count
(channel N: negative reports + M) robust where F = A + N is not? Same generator, seeds and learner as
sensitivity.py; subset of cells (kappa = 0.6, sil = 0). Usage: python3 -I explore_nonly.py <out.csv>"""
import importlib.util, itertools, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sens", os.path.join(HERE, "sensitivity.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
rows = []
for cell, (rho, kappa, q, pop, sil) in enumerate(itertools.product(*S.GRID.values())):
    if kappa != 0.6 or sil != 0.0:
        continue
    for rep in range(S.REPS):
        rng = np.random.default_rng(20261024 + 1000 * cell + rep)
        li = lambda y, fire, r: (r.choice((0, 1, 2, 4, 8), size=fire.shape) * fire).sum(1)  # noqa: E731
        ytr, Xtr = S.sample(S.N_TR, rho, q, pop, sil, rng, li)
        fx = {"N": lambda X: X["F_neg"][:, None], "U": lambda X: X["U"][:, None], "F": lambda X: np.c_[X["A"], X["F_neg"]]}
        models = {k: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1, random_state=0).fit(f(Xtr), ytr) for k, f in fx.items()}
        gaps = {"N": [], "F": []}
        for sign in (+1, -1):
            def ld(y, fire, r, s=sign):
                p = np.where(y == 1, 0.5 + s * kappa / 2, 0.5 - s * kappa / 2)
                return S.STORM * fire.sum(1) * (r.random(len(y)) < p)
            yte, Xte = S.sample(S.N_TE, rho, q, pop, sil, rng, ld)
            bu = S.brier(models["U"].predict_proba(fx["U"](Xte))[:, 1], yte)
            for k in gaps:
                gaps[k].append(S.brier(models[k].predict_proba(fx[k](Xte))[:, 1], yte) - bu)
        rows.append(dict(rho=rho, q=q, pop=pop, rep=rep, gap_N=max(gaps["N"]), gap_F=max(gaps["F"])))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False); print("done")
