"""Rows of Table 6 (tab:baselines) from results/est/baselines.csv and results/est/estimation_error.csv.
Usage: python3 tools/make_baseline_table.py -> results/est/baseline_rows.tex"""
import pandas as pd

b = pd.read_csv("results/est/baselines.csv"); e = pd.read_csv("results/est/estimation_error.csv"); DS = (1, 4, 16)
mb = b.groupby("D").mean(numeric_only=True); me = e.groupby("D").mean(numeric_only=True)
wins = lambda c: [int((z.W_minimax < z[c] - 1e-12).sum()) for D, z in b.groupby("D")]
f = lambda v: f"{v:.3f}"; rows = []
for name, col, src, w in [("Naive count", "W_naive", mb, True), ("Interval-midrange", "W_imid", mb, True), ("Interval-mean", "W_imean", mb, True),
                          ("TV-robust minimax", "W_tvrobust", me, False), ("Presence", "W_presence", mb, True),
                          ("Smoothed minimax (add 0.5)", "W_smoothed", me, False), ("Plug-in minimax", "W_minimax", mb, None),
                          ("Selection (minimax or presence)", "W_select", me, False)]:
    vals = " & ".join(f(src.loc[D, col]) for D in DS)
    tail = " & ".join(str(x) for x in wins(col)) if w else ("& &" if w is None else "-- & -- & --")
    rows.append(f"{name} & {vals} & {tail} \\\\" if w is not None else f"{name} & {vals} & & & \\\\")
open("results/est/baseline_rows.tex", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
