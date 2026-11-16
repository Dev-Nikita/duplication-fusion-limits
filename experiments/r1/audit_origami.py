"""R1: ORIGAMI data audit (state-machine-filtered CSVs). Streams each file; writes results/r1/audit_*.csv.

Usage: python3 experiments/r1/audit_origami.py data/origami/csv/case_study_csv results/r1/audit_case_study.csv
"""
import glob, os, re, sys
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
rows = []
files = sorted(glob.glob(os.path.join(src, "*.csv"))) if os.path.isdir(src) else [src]
for f in files:
    m = re.search(r"_(\d+)-Cores-all_(\d+)_", os.path.basename(f))
    cores, rate = int(m.group(1)), int(m.group(2))
    agg = {}
    ue_span = {}
    for ch in pd.read_csv(f, sep=";", chunksize=500_000,
                          dtype={"node": "category", "id": "category", "ue": "string"}):
        ch = ch.dropna(subset=["servicetime"])
        for (run, node), g in ch.groupby(["run", "node"], observed=True):
            a = agg.setdefault((run, node), [])
            a.append(g["servicetime"].to_numpy())
        for (run, ue), g in ch.groupby(["run", "ue"], observed=True):
            k = (run, ue)
            s, e = g["start"].min(), g["end"].max()
            if k in ue_span:
                ue_span[k] = (min(ue_span[k][0], s), max(ue_span[k][1], e))
            else:
                ue_span[k] = (s, e)
    spans = {}
    for (run, ue), (s, e) in ue_span.items():
        spans.setdefault(run, []).append(e - s)
    for (run, node), parts in agg.items():
        v = np.concatenate(parts)
        sp = np.asarray(spans.get(run, [np.nan]))
        rows.append(dict(file=os.path.basename(f), cores=cores, rate=rate, run=int(run), node=str(node),
                         n=len(v), st_median=float(np.median(v)), st_p95=float(np.percentile(v, 95)),
                         st_p99=float(np.percentile(v, 99)), n_ue=len(sp),
                         flow_median=float(np.nanmedian(sp)), flow_p95=float(np.nanpercentile(sp, 95))))
    print(os.path.basename(f), "runs", len({r for r, _ in agg}), flush=True)
os.makedirs(os.path.dirname(out), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out, index=False, mode="a", header=not os.path.exists(out))
