"""Estimation-error analysis of finite-cap fusion rules (protocol: docs/est-protocol.md). Analytical, existing data.
For each of the 108 outer folds (fold_laws.pkl) and D in {1,2,4,8,16}, five rules are fitted on the training
blocks and scored by their EXACT worst-case Brier risk over Delta_D on the held-out block law:
  plugin    : finite-cap minimax rule of the pooled training law
  presence  : Bayes rule of presence, fitted on the pooled training law (risk does not depend on duplication)
  smoothed  : finite-cap minimax rule of the add-0.5 smoothed training law (42 cells)
  tvrobust  : TV-robust finite-cap rule, eps = mean over training blocks of TV(block law, pooled training law)
  select    : plugin or presence, whichever has the lower mean exact worst-case risk over inner leave-one-
              training-block-out folds (each inner fit uses the remaining training blocks)
Usage: python3 -W ignore experiments/est/estimation_error.py <fold_laws.pkl> <out.csv>"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import finite_cap_cvx as C, finite_cap_solvers as S  # noqa: E402

DS = (1, 2, 4, 8, 16); ALPHA = 0.5


def norm(n):
    return n / n.sum()


def presence_rule(p, D):
    A = (D + 1) * (p.shape[1] - 1) + 1
    e0 = p[1, 0] / p[:, 0].sum() if p[:, 0].sum() > 0 else 0.5
    e1 = p[1, 1:].sum() / p[:, 1:].sum() if p[:, 1:].sum() > 0 else 0.5
    x = np.full(A, e1); x[0] = e0; return x


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def main():
    F = pickle.load(open(sys.argv[1], "rb")); rows = []
    for f in F:
        ntr, pte = f["n_train"], norm(f["n_test"]); ptr = norm(ntr)
        blocks = sorted(f["per_block"]); eps = float(np.mean([tv(norm(f["per_block"][b]), ptr) for b in blocks]))
        psm = norm(ntr + ALPHA)
        for D in DS:
            rules = {"plugin": C.ld_minimax(ptr, D)[1], "presence": presence_rule(ptr, D),
                     "smoothed": C.ld_minimax(psm, D)[1], "tvrobust": C.tv_robust_minimax(ptr, D, eps)[1]}
            inner = {"plugin": [], "presence": []}
            for j in blocks:
                nin = sum(f["per_block"][b] for b in blocks if b != j); pin = norm(nin); pj = norm(f["per_block"][j])
                if f["per_block"][j][1].sum() == 0 or f["per_block"][j][0].sum() == 0:
                    continue
                inner["plugin"].append(S.primal_value(C.ld_minimax(pin, D)[1], pj, D))
                inner["presence"].append(S.primal_value(presence_rule(pin, D), pj, D))
            pick = "plugin" if np.mean(inner["plugin"]) < np.mean(inner["presence"]) else "presence"
            rules["select"] = rules[pick]
            row = dict(dataset=f["dataset"], task=f["task"], held_block=f["held_block"], D=D, eps=eps, select_pick=pick,
                       inner_plugin=np.mean(inner["plugin"]), inner_presence=np.mean(inner["presence"]))
            for k, x in rules.items():
                row["W_" + k] = S.primal_value(x, pte, D)
            rows.append(row)
        print(f["dataset"], f["task"], f["held_block"], flush=True)
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
    main()
