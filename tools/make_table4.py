"""Rows of Table 4 (tab:copycap) from results/finite_cap/{summary,saturation_closed_form,li_vs_ld}.csv.
Usage: python3 tools/make_table4.py  -> results/finite_cap/table4_rows.tex (pasted verbatim into main_if.tex)"""
import pandas as pd

R = "results/finite_cap/"
s = pd.read_csv(R + "summary.csv"); sc = pd.read_csv(R + "saturation_closed_form.csv"); lv = pd.read_csv(R + "li_vs_ld.csv")
tn = {"amf_stall": "AMF", "ausf_stall": "AUSF", "net_delay": "Delay"}; dn = {"R11-control": "R11 ctrl", "R11-loaded": "R11 load"}
f = lambda v: f"{v:.3f}"; rows = []
for ds in ["R10", "R6", "R11-control", "R11-loaded"]:
    for t in ["amf_stall", "ausf_stall", "net_delay"]:
        z = s[(s.dataset == ds) & (s.task == t)].set_index("D"); v = sc[(sc.dataset == ds) & (sc.task == t)].D_sat_closed
        li = lv[(lv.dataset == ds) & (lv.task == t) & (lv.D == 4)].R_LI.mean()
        dt = f"{int(v.min())}" if v.min() == v.max() else f"{int(v.min())}--{int(v.max())}"
        rows.append(f"{dn.get(ds, ds)} & {tn[t]} & {f(z.RB_U[0])} & {f(z.RB_P[0])} & {f(z.R_star[1])} & {f(z.R_star[4])} & {f(li)} & "
                    f"{f(z.R_star[16])} & {dt} & {f(z.W_minimax[1])} / {f(z.W_naive[1])} & {f(z.W_minimax[16])} / {f(z.W_naive[16])} & "
                    f"{f(z.presence_test[0])} \\\\")
open(R + "table4_rows.tex", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
