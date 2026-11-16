"""R8 criteria (docs/r8-protocol.md). Unit = held-out block (n = 6); tasks restricted to the R6 gate."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(sys.argv[1]); g = pd.read_csv(sys.argv[2])
ok = g.groupby(["task", "held_block"]).passed.mean().reset_index(); ok = ok[ok.passed >= 2 / 3][["task", "held_block"]]
d = d.merge(ok)
DUP = ["am_ha_replica", "am_multiroute", "am_combined"]


def summ(a, b, sc, tasks=None, one_sided=False):
    x = d[d.scenario.isin(sc)] if tasks is None else d[d.scenario.isin(sc) & d.task.isin(tasks)]
    v = (x[f"{a}_brier"] - x[f"{b}_brier"]).groupby(x.held_block).mean(); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(.975, n - 1); r = dict(est=round(float(m), 4), t_ci95=[round(float(m - t * se), 4), round(float(m + t * se), 4)],
                                          positive_blocks=int((v > 0).sum()), per_block={int(k): round(float(z), 4) for k, z in v.items()})
    if one_sided:
        r["one_sided_upper95"] = round(float(m + stats.t.ppf(.95, n - 1) * se), 4)
    return r


out = {}
p1 = summ("augmented", "la_aug", DUP); p1["holds"] = bool(p1["t_ci95"][0] > 0 and p1["positive_blocks"] >= 5); out["R8_P1_la_aug_vs_augmented_dup"] = p1
p2 = summ("la_aug", "augmented", ["am_default"], one_sided=True); p2["holds"] = bool(p2["one_sided_upper95"] < 0.01); out["R8_P2_noninferiority_default"] = p2
out["per_scenario_la_vs_aug"] = {s: summ("augmented", "la_aug", [s]) for s in d.scenario.unique()}
out["per_scenario_a4_vs_aug"] = {s: summ("augmented", "a4", [s])["est"] for s in d.scenario.unique()}
out["per_scenario_gap_to_matched"] = {s: summ("augmented", "matched", [s])["est"] for s in d.scenario.unique()}
out["per_task_P1"] = {t: summ("augmented", "la_aug", DUP, tasks=[t])["est"] for t in sorted(d.task.unique())}
out["P1_excluding_reg_reject"] = summ("augmented", "la_aug", DUP, tasks=[t for t in d.task.unique() if t != "reg_reject"])
meths = ["stacking", "augmented", "a4", "la_aug", "presence", "matched"]
out["absolute_brier"] = {s: {m: round(float(d[d.scenario == s][f"{m}_brier"].mean()), 4) for m in meths} for s in d.scenario.unique()}
out["copies_per_window"] = d.groupby("scenario")[["n_copies", "n_invisible"]].mean().round(1).to_dict()
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), "r8_results.json"), "w"), indent=1, default=str)
