"""Rows of the saturation-stability table (tab:stability) from results/finite_cap/saturation_stability_{full,boot,lobo}.csv.
Usage: python3 tools/make_stability_table.py -> results/finite_cap/stability_rows.tex"""
import pandas as pd

R = "results/finite_cap/"
f = pd.read_csv(R + "saturation_stability_full.csv"); b = pd.read_csv(R + "saturation_stability_boot.csv")
l = pd.read_csv(R + "saturation_stability_lobo.csv")
tn = {"amf_stall": "AMF", "ausf_stall": "AUSF", "net_delay": "Delay"}; dn = {"R11-control": "R11 ctrl", "R11-loaded": "R11 load"}
rows = []
for ds in ["R10", "R6", "R11-control", "R11-loaded"]:
    for t in ["amf_stall", "ausf_stall", "net_delay"]:
        fz = f[(f.dataset == ds) & (f.task == t)].iloc[0]; z = b[(b.dataset == ds) & (b.task == t)]; lz = l[(l.dataset == ds) & (l.task == t)]
        q = lambda c: z[c].quantile([0.05, 0.95]).values
        ds_q = q("D_sat"); cells = [f"{int(fz.D_sat)}", f"{int(ds_q[0])}--{int(ds_q[1])}", f"{(z.D_sat == fz.D_sat).mean():.2f}",
                                     f"{int(lz.D_sat.min())}--{int(lz.D_sat.max())}"]
        for D in (1, 4, 16):
            v = q(f"hdelta_{D}"); cells.append(f"{fz[f'hdelta_{D}']:.3f} [{v[0]:.3f}, {v[1]:.3f}]")
        rows.append(f"{dn.get(ds, ds)} & {tn[t]} & " + " & ".join(cells) + " \\\\")
open(R + "stability_rows.tex", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
