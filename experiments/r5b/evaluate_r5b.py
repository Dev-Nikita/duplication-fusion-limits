"""R5b criteria (docs/r5b-protocol.md). Same resampling as R5: (task, held-out block) cell, seed-averaged;
hierarchical bootstrap over blocks then tasks."""
import os, sys, json
import numpy as np, pandas as pd

d = pd.read_csv(sys.argv[1]); g = pd.read_csv(sys.argv[2])
passed = g.groupby(["task", "held_block"]).passed.mean().reset_index()
fold_ok = passed[passed.passed >= 0.6][["task", "held_block"]]
task_ok = fold_ok.groupby("task").size(); tasks = list(task_ok[task_ok >= 2].index)
d = d.merge(fold_ok, on=["task", "held_block"]); d = d[d.task.isin(tasks)]
rng = np.random.default_rng(11)


def cell(D, h, col):
    return d[(d.D == D) & (d.h == h)].groupby(["task", "held_block"])[col].mean()


def hboot(series, B=20000):
    s = series.reset_index(); s.columns = ["task", "block", "v"]; blocks = s.block.unique(); bs = []
    for _ in range(B):
        vals = []
        for b in rng.choice(blocks, len(blocks), replace=True):
            sb = s[s.block == b]; vals += list(sb.v.to_numpy()[rng.integers(0, len(sb), len(sb))])
        bs.append(np.mean(vals))
    return float(s.v.mean()), np.percentile(bs, [2.5, 97.5]).round(4).tolist(), float(np.percentile(bs, 95)), \
        s.groupby("block").v.mean().round(4).to_dict()


def diff(D, h, a, b="a4"):
    return cell(D, h, f"{a}_brier") - cell(D, h, f"{b}_brier")


out = {"tasks_passing_gate": tasks, "gate": passed.to_dict("records")}
for name, (D, h, base) in {"B1_vs_presence": (8, .5, "presence"), "B2_vs_augmented": (8, .5, "augmented"),
                           "B3_vs_stacking": (8, .5, "stacking")}.items():
    est, ci, _, pb = hboot(diff(D, h, base))
    out[name] = dict(est=est, ci95=ci, per_block=pb, holds=bool(ci[0] > 0 and all(v > 0 for v in pb.values())))
est, ci, up, pb = hboot(diff(1, .5, "a4", "stacking"))
out["B4_noninferiority_D1"] = dict(est=est, ci95=ci, one_sided_upper95=round(up, 4), per_block=pb, holds=bool(up < 0.01))
c3 = (cell(16, 0.0, "a4_brier") - cell(0, 0.0, "a4_brier")).abs()
out["B5_invariance"] = dict(max_abs=float(c3.max()), holds=bool(c3.max() < 0.005))
for name, (D, h, base) in {"S1_vs_augmented_unseenD16": (16, .5, "augmented"), "S2_vs_matched_oracle": (8, .5, "matched"),
                           "S3_vs_log1p": (8, .5, "log1p"), "S4_vs_pact_declared": (8, .5, "pact_declared"),
                           "S5_vs_tb_loglinear": (8, .5, "tb_loglinear"), "S6_vs_naive_bayes": (8, .5, "naive_bayes"),
                           "S7_vs_presence_D1": (1, .5, "presence")}.items():
    est, ci, _, pb = hboot(diff(D, h, base)); out[name] = dict(est=est, ci95=ci, per_block=pb)
e, ci, _, pb = hboot(cell(8, .5, "dec_err") - cell(8, .5, "a4_err")); out["S8_decided_minus_point_error"] = dict(est=e, ci95=ci)
tab = {m: {f"D{D}_h{h}": round(float(cell(D, h, f"{m}_brier").mean()), 4) for D, h in [(1, .5), (8, .5), (16, .5), (8, 1.0)]}
       for m in ["stacking", "a4", "presence", "log1p", "augmented", "matched", "naive_bayes", "tb_loglinear", "pact_declared", "pact_true"]}
out["absolute_brier"] = tab
if "--quiet" not in sys.argv:
    print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), "r5b_confirmatory.json"), "w"), indent=1, default=str)
print("written")
