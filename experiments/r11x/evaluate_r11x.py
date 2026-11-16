"""R11-X frozen evaluation. Usage: python3 -I evaluate_r11x.py <alarm_cells.csv> <full_cells.csv> <out.json>
Per held-out block b (unit of inference), primary tasks amf_stall, ausf_stall, net_delay:
  G_ch(ref, b) = mean over tasks of max over the LD pair of [Brier(ref) - Brier(gbdt_unit)] in channel ch.
X1 (lineage matters without the denominator): G_alarm(gbdt_aug) - DELTA has one-sided 95 % lower bound > 0
   and >= 5/6 of blocks have G_alarm(gbdt_aug) > DELTA.
X2 (lineage-separated features recover it): G_alarm(gbdt_la) one-sided 95 % upper bound < DELTA.
X3 (the denominator makes lineage redundant): G_full(gbdt_aug) one-sided 95 % upper bound < DELTA.
DELTA = 0.005 (R10 negligibility margin)."""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
DELTA, LDM, TASKS = 0.005, ("storm_inc", "storm_ben"), ("amf_stall", "ausf_stall", "net_delay")


def G(cells, ref):
    c = cells[cells.task.isin(TASKS)].copy(); c["d"] = c[ref] - c.gbdt_unit
    s = c.groupby(["task", "held_block", "mech"]).d.mean().reset_index()
    return s[s.mech.isin(LDM)].groupby(["task", "held_block"]).d.max().groupby("held_block").mean()


def summ(x):
    n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.95, n - 1)
    return dict(est=round(m, 5), lb95=round(m - t * se, 5), ub95=round(m + t * se, 5), n=n,
                above_delta=int((x > DELTA).sum()), per_block={int(k): round(float(v), 5) for k, v in x.items()})


a, f = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
ga, gl, gf = summ(G(a, "gbdt_aug")), summ(G(a, "gbdt_la")), summ(G(f, "gbdt_aug"))
res = dict(alarm_gbdt_aug=ga, alarm_gbdt_la=gl, full_gbdt_aug=gf,
           alarm_mix_oracle=summ(G(a, "gbdt_mix_oracle")), alarm_logistic=summ(G(a, "augmented")),
           X1=bool(ga["lb95"] > DELTA and ga["above_delta"] >= int(np.ceil(5 / 6 * ga["n"]))),
           X2=bool(gl["ub95"] < DELTA), X3=bool(gf["ub95"] < DELTA))
json.dump(res, open(sys.argv[3], "w"), indent=1)
print(json.dumps({k: (v if isinstance(v, bool) else {kk: v[kk] for kk in ("est", "lb95", "ub95", "above_delta", "n")}) for k, v in res.items()}, indent=1))
