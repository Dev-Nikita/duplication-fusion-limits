"""EXPLORATORY critical check (2026-10-24): is the lineage-free loss mostly a training-support artefact?
Same generator as sensitivity.py (known M, q = 0, sil = 0). Training augmentation D in {0,1,2,4,8} (as frozen) vs
a wide support D in {0,1,2,4,8,16,32} that covers the test storm size (16). Channels: A (firing counts),
F (A + negatives), N (negatives only). Reference U. Usage: python3 -I explore_support.py <out.csv>"""
import importlib.util, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sens", os.path.join(HERE, "sensitivity.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
FX = {"A": lambda X: X["A"][:, None], "F": lambda X: np.c_[X["A"], X["F_neg"]], "N": lambda X: X["F_neg"][:, None],
      "logA": lambda X: np.log1p(X["A"])[:, None], "U": lambda X: X["U"][:, None]}
rows = []
for support, Ds in (("narrow", (0, 1, 2, 4, 8)), ("wide", (0, 1, 2, 4, 8, 16, 32))):
    for rho in (0.1, 0.3, 0.6, 0.9):
        for kappa in (0.0, 0.3, 0.6):
            for rep in range(5):
                rng = np.random.default_rng(777 + rep + int(rho * 100) * 10 + int(kappa * 10))
                li = lambda y, fire, r, Ds=Ds: (r.choice(Ds, size=fire.shape) * fire).sum(1)  # noqa: E731
                ytr, Xtr = S.sample(S.N_TR, rho, 0.0, "known", 0.0, rng, li)
                m = {k: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1, random_state=0).fit(f(Xtr), ytr) for k, f in FX.items()}
                gaps = {k: [] for k in FX if k != "U"}
                for sign in (+1, -1):
                    def ld(y, fire, r, s=sign):
                        p = np.where(y == 1, 0.5 + s * kappa / 2, 0.5 - s * kappa / 2)
                        return S.STORM * fire.sum(1) * (r.random(len(y)) < p)
                    yte, Xte = S.sample(S.N_TE, rho, 0.0, "known", 0.0, rng, ld)
                    bu = S.brier(m["U"].predict_proba(FX["U"](Xte))[:, 1], yte)
                    for k in gaps:
                        gaps[k].append(S.brier(m[k].predict_proba(FX[k](Xte))[:, 1], yte) - bu)
                rows.append(dict(support=support, rho=rho, kappa=kappa, rep=rep, **{f"gap_{k}": max(v) for k, v in gaps.items()}))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False); print("done")
