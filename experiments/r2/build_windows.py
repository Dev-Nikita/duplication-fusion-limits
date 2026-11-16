"""R2a: reduce ORIGAMI case-study CSVs to window-level statistics with message-set lineage keys.

Window = WIN consecutive UEs of one run, ordered by the UE's first event (rate-independent unit).
For every (window, node) we keep n events and several statistics of the SAME message set
(mean, p90, share of events above the node's global median): these are distinct "readings" of one
origin (the node's message stream in that window). Lineage is therefore known by construction:
same (run, window, node) => same origin; different node => disjoint message sets.

Usage: python3 experiments/r2/build_windows.py <csv> <out.csv> [WIN]
"""
import os, re, sys
import numpy as np
import pandas as pd

f, out = sys.argv[1], sys.argv[2]
WIN = int(sys.argv[3]) if len(sys.argv) > 3 else 50
m = re.search(r"_(\d+)-Cores-all_(\d+)_", os.path.basename(f))
cores, rate = int(m.group(1)), int(m.group(2))
d = pd.read_csv(f, sep=";", usecols=["node", "servicetime", "run", "start", "end", "ue"],
                dtype={"node": "category", "ue": "string"}).dropna(subset=["servicetime"])
d = d[d.node != "none"]
rows = []
for run, g in d.groupby("run", observed=True):
    first = g.groupby("ue", observed=True)["start"].min().sort_values()
    win_of = pd.Series(np.arange(len(first)) // WIN, index=first.index)
    g = g.assign(win=g["ue"].map(win_of))
    span = g.groupby("ue", observed=True).agg(s=("start", "min"), e=("end", "max"))
    span["win"] = win_of.reindex(span.index).values
    span["flow"] = span.e - span.s
    for w, gw in g.groupby("win"):
        n_ue = int((win_of == w).sum())
        if n_ue < WIN // 2:
            continue
        fl = span.loc[span.win == w, "flow"].to_numpy()
        base = dict(cores=cores, rate=rate, run=int(run), win=int(w), n_ue=n_ue,
                    t_end=float(gw["end"].max()),
                    flow_p90=float(np.percentile(fl, 90)), flow_mean=float(fl.mean()))
        for node, gn in gw.groupby("node", observed=True):
            v = gn["servicetime"].to_numpy()
            if len(v) == 0:
                continue
            base[f"{node}_n"] = len(v)
            base[f"{node}_mean"] = float(v.mean())
            base[f"{node}_p90"] = float(np.percentile(v, 90))
            base[f"{node}_max"] = float(v.max())
        rows.append(base)
df = pd.DataFrame(rows)
df.to_csv(out, index=False, mode="a", header=not os.path.exists(out))
print(os.path.basename(f), len(df), "windows")
