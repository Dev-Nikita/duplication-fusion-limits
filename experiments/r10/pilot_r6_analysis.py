import sys, pandas as pd, numpy as np
P = sys.argv[1]
out = []
for m in (2, 4, 8, 20):
    c = pd.read_csv(f"{P}/m{m}/r10_cells.csv"); g = pd.read_csv(f"{P}/m{m}/r10_gate.csv")
    c = c[c.arm != "ERR"]
    for task in ("amf_stall", "ausf_stall", "net_delay"):
        s = c[c.task == task].groupby(["held_block", "mech"])[["augmented", "gbdt_aug", "gbdt_unit", "gbdt_mix_oracle"]].mean().reset_index()
        ld = s[s.mech.isin(["storm_inc", "storm_ben"])].copy(); li = s[~s.mech.isin(["storm_inc", "storm_ben"])]
        ld["aug_gap"] = ld.gbdt_aug - ld.gbdt_unit; ld["mix_gap"] = ld.gbdt_mix_oracle - ld.gbdt_unit
        wc = ld.groupby("held_block")[["aug_gap", "mix_gap"]].max()
        li_gap = (li.gbdt_aug - li.gbdt_unit).groupby(li.held_block).mean()
        out.append(dict(m=m, task=task, auc=round(g[g.task == task].auc_cal.mean(), 3), brier_unit=round(s.gbdt_unit.mean(), 4),
                        LD_aug_gap=round(wc.aug_gap.mean(), 4), LD_aug_pos=f"{(wc.aug_gap > 0).sum()}/{len(wc)}",
                        LD_mix_gap=round(wc.mix_gap.mean(), 4), LD_mix_pos=f"{(wc.mix_gap > 0).sum()}/{len(wc)}",
                        LI_gap=round(li_gap.mean(), 4)))
print(pd.DataFrame(out).to_string(index=False))
