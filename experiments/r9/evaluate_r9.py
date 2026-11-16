import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(sys.argv[1])

def summ(x, a, b):
    v = (x[a] - x[b]).groupby(x.held_block).mean(); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(.975, n - 1)
    return dict(est=round(float(m), 4), t_ci95=[round(float(m - t * se), 4), round(float(m + t * se), 4)], positive_blocks=int((v > 0).sum()))

g = d[d.part == "gbdt"]
out = {"G1_gbdt_aug_minus_la_aug": summ(g, "gbdt_aug", "la_aug"), "G2_gbdt_aug_minus_gbdt_la": summ(g, "gbdt_aug", "gbdt_la"),
       "gbdt_aug_minus_augmented": summ(g, "gbdt_aug", "augmented"),
       "absolute": g[["augmented", "la_aug", "gbdt_aug", "gbdt_la"]].mean().round(4).to_dict(),
       "G2_per_task": {t: summ(g[g.task == t], "gbdt_aug", "gbdt_la")["est"] for t in sorted(g.task.unique())}}
for kind in ("false_split", "false_merge"):
    x = d[d.part == kind]
    out[kind] = {str(e): summ(x[x.eps == e], "augmented", "la_aug") for e in sorted(x.eps.unique())}
out["mnar"] = summ(d[d.part == "mnar"], "augmented", "la_aug")
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), "r9_results.json"), "w"), indent=1)
