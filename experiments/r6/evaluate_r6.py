"""R6 criteria (docs/r6-protocol.md). Primary unit: held-out BLOCK (6 independent values; tasks and seeds
averaged within block). Interval: t-interval over blocks + sign count; block bootstrap as secondary."""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(sys.argv[1]); g = pd.read_csv(sys.argv[2])
passed = g.groupby(["task", "held_block"]).passed.mean().reset_index()
fold_ok = passed[passed.passed >= 2 / 3][["task", "held_block"]]           # >= 2 of 3 seeds
task_ok = fold_ok.groupby("task").size(); tasks = list(task_ok[task_ok >= 4].index)   # >= 4 of 6 folds
d = d.merge(fold_ok, on=["task", "held_block"]); d = d[d.task.isin(tasks)]
UNSEEN = ["burst", "source", "incident", "storm", "mixed"]
rng = np.random.default_rng(13)


def per_block(mechs, h, a, b):
    x = d[d.mech.isin(mechs) & (d.h == h)].copy(); x["v"] = x[f"{a}_brier"] - x[f"{b}_brier"]
    return x.groupby("held_block").v.mean()


def summarize(s, ni_margin=None):
    v = s.to_numpy(); n = len(v); m = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(n))
    t = stats.t.ppf(0.975, n - 1); t1 = stats.t.ppf(0.95, n - 1)
    bs = [rng.choice(v, n, replace=True).mean() for _ in range(20000)]
    out = dict(est=round(m, 4), t_ci95=[round(m - t * se, 4), round(m + t * se, 4)],
               boot_ci95=np.percentile(bs, [2.5, 97.5]).round(4).tolist(), positive_blocks=int((v > 0).sum()), n_blocks=n,
               per_block={int(k): round(float(x), 4) for k, x in s.items()})
    if ni_margin is not None:
        out["one_sided_upper95"] = round(m + t1 * se, 4)
    return out


out = {"tasks_passing_gate": tasks}
p1 = summarize(per_block(UNSEEN, 0.5, "augmented", "la_aug"))
p1["holds"] = bool(p1["t_ci95"][0] > 0 and p1["positive_blocks"] >= p1["n_blocks"] - 1); out["P1_la_aug_vs_augmented_unseen"] = p1
p2 = summarize(per_block(["iid"], 0.5, "la_aug", "augmented"), ni_margin=0.01); p2["holds"] = bool(p2["one_sided_upper95"] < 0.01)
out["P2_noninferiority_iid"] = p2
for name, (mechs, a, b) in {"S1_a4_vs_augmented_unseen": (UNSEEN, "augmented", "a4"),
                            "S2_gen_bma_vs_augmented_unseen": (UNSEEN, "augmented", "gen_bma"),
                            "S3_la_aug_vs_gen_bma_unseen": (UNSEEN, "gen_bma", "la_aug"),
                            "S4_la_aug_vs_matched_unseen": (UNSEEN, "matched", "la_aug"),
                            "S5_a4_incl_vs_a4_unseen": (UNSEEN, "a4_incl", "a4"),
                            "S6_la_aug_vs_stacking_unseen": (UNSEEN, "stacking", "la_aug")}.items():
    out[name] = summarize(per_block(mechs, 0.5, a, b))
out["per_mechanism_la_aug_vs_augmented"] = {m: summarize(per_block([m], 0.5, "augmented", "la_aug")) for m in ["iid"] + UNSEEN}
out["availability_curve_la_aug_vs_augmented"] = {str(h): summarize(per_block(UNSEEN, h, "augmented", "la_aug")) for h in [0, .25, .5, .75, 1.0]}
x = d[d.mech.isin(UNSEEN) & (d.h == .5)]; x = x.assign(v=x.dec_err - x.a4_err)
out["S7_decided_minus_point_error_unseen"] = summarize(x.groupby("held_block").v.mean())
meths = ["stacking", "augmented", "a4", "a4_incl", "la_aug", "gen_bma", "naive_bayes", "tb_loglinear", "pact_declared", "presence", "matched"]
out["absolute_brier_h0.5"] = {mech: {m: round(float(d[(d.mech == mech) & (d.h == .5)][f"{m}_brier"].mean()), 4) for m in meths}
                              for mech in ["iid"] + UNSEEN}
if "--quiet" not in sys.argv:
    print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), "r6_confirmatory.json"), "w"), indent=1, default=str)
print("written")
