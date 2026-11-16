"""R10 criteria (docs/r10-protocol.md). Unit = held-out block (n = 6); primary tasks: amf_stall, ausf_stall, net_delay."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(sys.argv[1]); g = pd.read_csv(sys.argv[2])
g["passed"] = g.auc_cal >= 0.65
ok = g.groupby(["task", "held_block"]).passed.mean().reset_index(); ok = ok[ok.passed >= 2 / 3][["task", "held_block"]]
d = d.merge(ok)
PRIMARY = ["amf_stall", "ausf_stall", "net_delay"]


def tsum(v, one_sided=False):
    n = len(v); m = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(n)); t = stats.t.ppf(.975, n - 1)
    r = dict(est=round(m, 4), t_ci95=[round(m - t * se, 4), round(m + t * se, 4)], positive_blocks=int((v > 0).sum()), n_blocks=n,
             per_block={int(k): round(float(x), 4) for k, x in v.items()})
    if one_sided:
        r["one_sided_upper95"] = round(m + stats.t.ppf(.95, n - 1) * se, 4)
    return r


def ld_worst(x, method):
    w = x[x.arm == "LD"].groupby(["held_block", "task", "seed", "mech"])[method].mean().groupby(["held_block", "task", "seed"]).max()
    return w.groupby(["held_block", "task"]).mean()


def by_block(series):
    return series.groupby(level=0).mean()


out = {}
for label, tasks in (("primary", PRIMARY), ("reg_reject", ["reg_reject"])):
    x = d[d.task.isin(tasks)]
    if x.empty:
        continue
    res = {}
    # P1: worst case over the LD pair: X-only mixture oracle vs exact-lineage unit model
    res["P1_ld_worst_mixoracle_minus_unit"] = tsum(by_block(ld_worst(x, "gbdt_mix_oracle") - ld_worst(x, "gbdt_unit")))
    res["P1_ld_worst_gbdtaug_minus_unit"] = tsum(by_block(ld_worst(x, "gbdt_aug") - ld_worst(x, "gbdt_unit")))
    # P2: LI arm, gbdt without lineage non-inferior to unit model (margin 0.005)
    li = x[x.arm == "LI"]; v = (li.gbdt_aug - li.gbdt_unit).groupby(li.held_block).mean()
    res["P2_li_gbdtaug_minus_unit"] = tsum(v, one_sided=True); res["P2_li_gbdtaug_minus_unit"]["holds"] = bool(res["P2_li_gbdtaug_minus_unit"]["one_sided_upper95"] < 0.005)
    # P3: ERR arm, split harm grows with storm size; merge harm does not
    e = x[x.arm == "ERR"].pivot_table(index=["held_block", "task", "seed"], columns="mech", values="harm")
    sp = (e["err_split_16"] - e["err_split_4"]).groupby(level=0).mean(); mg = (e["err_merge_16"] - e["err_merge_4"]).groupby(level=0).mean()
    res["P3_split_growth"] = tsum(sp); res["P3_merge_growth"] = tsum(mg); res["P3_split_minus_merge_growth"] = tsum(sp - mg)
    res["P1_holds"] = bool(res["P1_ld_worst_mixoracle_minus_unit"]["t_ci95"][0] > 0 and res["P1_ld_worst_mixoracle_minus_unit"]["positive_blocks"] >= 5)
    res["P3_holds"] = bool(res["P3_split_minus_merge_growth"]["t_ci95"][0] > 0 and res["P3_split_minus_merge_growth"]["positive_blocks"] >= 5)
    meths = ["augmented", "la_aug", "gbdt_aug", "gbdt_la", "gbdt_unit", "gbdt_mix_oracle"]
    res["absolute_brier"] = {m: {mech: round(float(x[x.mech == mech][m].mean()), 4) for mech in x[x.arm != "ERR"].mech.unique()} for m in meths}
    out[label] = res
if "--quiet" not in sys.argv:
    print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), "r10_results.json"), "w"), indent=1)
print("written")
