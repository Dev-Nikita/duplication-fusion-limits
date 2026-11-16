"""R6-X criteria (docs/r6x-protocol.md). Unit = held-out block; tasks restricted to the R6 gate."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(sys.argv[1]); g = pd.read_csv(sys.argv[2])
ok = g.groupby(["task", "held_block"]).passed.mean().reset_index(); ok = ok[ok.passed >= 2 / 3][["task", "held_block"]]
d = d.merge(ok)
U = ["burst", "source", "incident", "storm", "mixed"]


def summ(a, b, mechs=U, tasks=None):
    x = d[d.mech.isin(mechs)] if tasks is None else d[d.mech.isin(mechs) & d.task.isin(tasks)]
    v = (x[f"{a}_brier"] - x[f"{b}_brier"]).groupby(x.held_block).mean(); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(.975, n - 1)
    return dict(est=round(float(m), 4), t_ci95=[round(float(m - t * se), 4), round(float(m + t * se), 4)], positive_blocks=int((v > 0).sum()),
                per_block={int(k): round(float(z), 4) for k, z in v.items()})


out = {"check_reproduces_R6_P1": summ("augmented", "la_aug"),
       "A1_placebo_minus_la": summ("la_placebo", "la_aug"),
       "A2_capacity_minus_la": summ("aug_cap", "la_aug"),
       "A3_tuned_aug_minus_tuned_la": summ("aug_tuned", "la_tuned"),
       "S_placebo_minus_augmented": summ("la_placebo", "augmented"),
       "S_capacity_minus_augmented": summ("aug_cap", "augmented")}
for a, b, name in [("la_placebo", "la_aug", "A1"), ("aug_cap", "la_aug", "A2"), ("aug_tuned", "la_tuned", "A3")]:
    out[f"{name}_per_task"] = {t: summ(a, b, tasks=[t])["est"] for t in sorted(d.task.unique())}
    out[f"{name}_excluding_reg_reject"] = summ(a, b, tasks=[t for t in d.task.unique() if t != "reg_reject"])
for k in ("A1_placebo_minus_la", "A2_capacity_minus_la", "A3_tuned_aug_minus_tuned_la"):
    v = out[k]; v["holds"] = bool(v["t_ci95"][0] > 0 and v["positive_blocks"] >= 5)
out["C_chosen"] = {"aug": d.C_aug.value_counts().to_dict(), "la": d.C_la.value_counts().to_dict()}
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), "r6x_results.json"), "w"), indent=1, default=str)
