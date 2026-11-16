"""Execution wrapper for sensitivity.py (amendment S2): identical cells, seeds, model and learner; results are
appended per cell so a run can be resumed. Usage: python3 -I run_resumable.py <out.csv>"""
import importlib.util, itertools, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sens", os.path.join(HERE, "sensitivity.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
out = sys.argv[1]
done = set(pd.read_csv(out).cell.unique()) if os.path.exists(out) else set()
for cell, (rho, kappa, q, pop, sil) in enumerate(itertools.product(*S.GRID.values())):
    if cell in done:
        continue
    rows = []
    for rep in range(S.REPS):
        rng = np.random.default_rng(20261024 + 1000 * cell + rep)
        li = lambda y, fire, r: (r.choice((0, 1, 2, 4, 8), size=fire.shape) * fire).sum(1)  # noqa: E731
        ytr, Xtr = S.sample(S.N_TR, rho, q, pop, sil, rng, li)
        models = {ch: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1, random_state=0)
                  .fit(S.feats(Xtr, ch), ytr) for ch in ("A", "F", "A+U", "U")}
        gaps = {ch: [] for ch in ("A", "F", "A+U")}
        for sign in (+1, -1):
            def ld(y, fire, r, s=sign):
                p = np.where(y == 1, 0.5 + s * kappa / 2, 0.5 - s * kappa / 2)
                return S.STORM * fire.sum(1) * (r.random(len(y)) < p)
            yte, Xte = S.sample(S.N_TE, rho, q, pop, sil, rng, ld)
            bu = S.brier(models["U"].predict_proba(S.feats(Xte, "U"))[:, 1], yte)
            for ch in gaps:
                gaps[ch].append(S.brier(models[ch].predict_proba(S.feats(Xte, ch))[:, 1], yte) - bu)
        rows.append(dict(cell=cell, rho=rho, kappa=kappa, q=q, pop=pop, sil=sil, rep=rep,
                         **{f"gap_{ch}": max(v) for ch, v in gaps.items()}))
    pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    print(cell, flush=True)
print("SIM_COMPLETE")
