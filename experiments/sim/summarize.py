"""Summarise the sensitivity map (docs/sim-protocol.md). Per cell: mean gap over 5 reps with a t-interval;
class 'works' if upper < DELTA, 'fails' if lower > DELTA, else 'undetermined'. DELTA = 0.005.
Usage: python3 -I summarize.py <sensitivity.csv>"""
import sys
import numpy as np, pandas as pd
DELTA, T = 0.005, 2.776
d = pd.read_csv(sys.argv[1]); keys = ["rho", "kappa", "q", "pop", "sil"]
g = d.groupby(keys).agg(**{f"{c}_{s}": (f"gap_{c}", s) for c in ("A", "F", "A+U") for s in ("mean", "std")}).reset_index()
for c in ("A", "F", "A+U"):
    lo = g[f"{c}_mean"] - T * g[f"{c}_std"] / np.sqrt(5); hi = g[f"{c}_mean"] + T * g[f"{c}_std"] / np.sqrt(5)
    g[f"{c}_cls"] = np.where(hi < DELTA, "works", np.where(lo > DELTA, "fails", "undet"))
pd.set_option("display.width", 200)
print("== class counts (192 cells)")
for c in ("A", "F", "A+U"):
    print(c, g[f"{c}_cls"].value_counts().to_dict())
print("\n== E1: A gap by kappa (mean over cells)"); print(g.groupby("kappa")["A_mean"].agg(["mean", "min", "max"]).round(4))
base = g[(g["pop"] == "known") & (g.sil == 0)]
print("\n== E2: F gap by q (known pop, sil=0), by kappa=0.6"); print(base[base.kappa == 0.6].pivot_table(index="q", columns="rho", values="F_mean").round(4))
print("\n== E3: F gap, pop unknown vs known (sil=0, q=0, kappa=0.6) by rho")
print(g[(g.sil == 0) & (g.q == 0) & (g.kappa == 0.6)].pivot_table(index="pop", columns="rho", values="F_mean").round(4))
print("\n== E4: F gap, sil 0 vs 0.5 (known pop, kappa=0.6) by q x rho")
print(g[(g["pop"] == "known") & (g.kappa == 0.6)].pivot_table(index=["sil", "q"], columns="rho", values="F_mean").round(4))
print("\n== E5: A+U gap"); print(g["A+U_mean"].describe().round(4))
print("\n== A gap (known pop, sil=0, q=0) kappa x rho"); print(base[base.q == 0].pivot_table(index="kappa", columns="rho", values="A_mean").round(4))
print("\n== F class map, kappa=0.6"); print(g[g.kappa == 0.6].pivot_table(index=["pop", "sil", "q"], columns="rho", values="F_cls", aggfunc="first"))
g.to_csv(sys.argv[1].replace(".csv", "_cells.csv"), index=False)
