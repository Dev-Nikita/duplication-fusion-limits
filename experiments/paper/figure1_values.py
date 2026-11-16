"""Values of Figure 1 (and of the count/presence columns of Table 2 and the X3 row of Table 1), computed from the
frozen cell files; nothing is transcribed by hand. Brier loss relative to the learned unique-count fuser
(gbdt_unit). For every (task, held-out block): mean over seeds per LD mechanism, worst of the two LD mechanisms;
then mean over tasks and blocks.
  count     = gbdt_aug   (firing counts, multiplicity augmentation), positive-only channel
  presence  = gbdt_dedup (presence features; identical under every mechanism)
  negatives = gbdt_aug   on the channel with non-firing reports (R11-X X3; not available for R11)
Usage: python3 experiments/paper/figure1_values.py -> results/editorial/fig1_channel_gaps.csv"""
import os
import numpy as np, pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results")


def worst_ld(path, col, cond=None):
    d = pd.read_csv(path); d = d[d.arm == "LD"]
    if cond:
        d = d[d["cond"] == cond]
    g = d.assign(gap=d[col] - d.gbdt_unit).groupby(["task", "held_block", "mech"]).gap.mean()
    return float(g.groupby(["task", "held_block"]).max().mean())


rows = []
for label, alarm, full, cond in [("R10", "r11x_rep/r10_alarm", "r11x_rep/r10_full", None),
                                 ("R6", "r11x_rep/r6_alarm", "r11x_rep/r6_full", None),
                                 ("R11 control", "r11x", None, "control"), ("R11 loaded", "r11x", None, "loaded")]:
    a = f"{R}/{alarm}/r11_cells.csv"
    rows.append(dict(setting=label, count=worst_ld(a, "gbdt_aug", cond), presence=worst_ld(a, "gbdt_dedup", cond),
                     negatives=worst_ld(f"{R}/{full}/r11_cells.csv", "gbdt_aug") if full else np.nan))
os.makedirs(f"{R}/editorial", exist_ok=True)
df = pd.DataFrame(rows); df.to_csv(f"{R}/editorial/fig1_channel_gaps.csv", index=False); print(df.round(4).to_string(index=False))
