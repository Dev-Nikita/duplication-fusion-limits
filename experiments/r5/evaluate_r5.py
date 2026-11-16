"""Confirmatory criteria for R5 (docs/r5-protocol.md). Unit of resampling: (task, held-out block) cell,
averaged over the 5 seeds; hierarchical bootstrap = resample blocks, then tasks within block."""
import os, sys, json
import numpy as np, pandas as pd

d = pd.read_csv(sys.argv[1]); g = pd.read_csv(sys.argv[2])
passed = g.groupby(["task", "held_block"]).passed.mean().reset_index()
fold_ok = passed[passed.passed >= 0.6][["task", "held_block"]]        # >= 3 of 5 seeds pass
task_ok = fold_ok.groupby("task").size()
tasks = list(task_ok[task_ok >= 2].index)                              # >= 2 of 3 held-out blocks
d = d.merge(fold_ok, on=["task", "held_block"])
d = d[d.task.isin(tasks)]
rng = np.random.default_rng(7)


def cell(D, h, col):
    x = d[(d.D == D) & (d.h == h)].groupby(["task", "held_block"])[col].mean()
    return x


def hboot(series, B=20000):
    s = series.reset_index(); s.columns = ["task", "block", "v"]
    blocks = s.block.unique(); est = s.v.mean(); bs = []
    for _ in range(B):
        bb = rng.choice(blocks, len(blocks), replace=True)
        vals = []
        for b in bb:
            sb = s[s.block == b]; vals += list(sb.v.to_numpy()[rng.integers(0, len(sb), len(sb))])
        bs.append(np.mean(vals))
    return float(est), np.percentile(bs, [2.5, 97.5]).round(4).tolist(), float(np.percentile(bs, 95))


out = {"tasks_passing_gate": tasks, "gate": passed.to_dict("records")}
c1 = cell(8, 0.5, "stacking_brier") - cell(8, 0.5, "a4_brier")
est, ci, _ = hboot(c1); per_block = c1.groupby(level=1).mean().round(4).to_dict()
out["C1"] = dict(est=est, ci95=ci, per_block=per_block, holds=bool(ci[0] > 0 and all(v > 0 for v in per_block.values())))
c2 = cell(1, 0.5, "a4_brier") - cell(1, 0.5, "stacking_brier")
est, ci, up95 = hboot(c2)
out["C2"] = dict(est=est, ci95=ci, one_sided_upper95=round(up95, 4), holds=bool(up95 < 0.01))
c3 = (cell(16, 0.0, "a4_brier") - cell(0, 0.0, "a4_brier")).abs()
out["C3"] = dict(max_abs=float(c3.max()), holds=bool(c3.max() < 0.005))
c4 = cell(8, 0.5, "cert_err") - cell(8, 0.5, "a4_err")
est, ci, _ = hboot(c4); out["C4"] = dict(est=est, ci95=ci, holds=bool(ci[1] < 0))
for b in ("pact_declared", "tb_loglinear", "geo_pool", "naive_bayes", "gen_bma"):
    c = cell(8, 0.5, f"{b}_brier") - cell(8, 0.5, "a4_brier")
    est, ci, _ = hboot(c); out[f"C5_vs_{b}"] = dict(est=est, ci95=ci, holds=bool(ci[0] > 0))
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), "r5_confirmatory.json"), "w"), indent=1, default=str)
