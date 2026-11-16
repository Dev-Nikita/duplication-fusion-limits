"""R11 technical pilot check: does the load sidecar reproduce the R6-type disturbance without breaking the core?
Only the manipulation check below is computed: no lineage model is fitted and no P1-type contrast is computed.
Usage: python3 -I pilot_check.py <episodes_root> <units.csv>
Pre-stated rule for W: W is the smallest pilot level such that
  (a) the 90th percentile of the healthy-episode median UE latency is >= 0.87 s (lower edge of the R6 blocks
      10-12 range; R10 gave 0.35-0.66 s),
  (b) every NF is alive after every episode of that level,
  (c) no healthy/stall episode has > 50 % rejected UEs, and
  (d) the stall classes keep a higher median latency than healthy at that level.
If no level satisfies (a)-(d): NO-GO, i.e. the load does not reproduce the disturbance and R11 is not run."""
import glob, json, os, sys
import numpy as np, pandas as pd

root, units = sys.argv[1], sys.argv[2]
meta = {}
for p in glob.glob(os.path.join(root, "episodes", "ep_*", "meta.json")):
    m = json.load(open(p))
    if m.get("block") is not None and 900 <= m["block"] < 910:
        meta[m["episode"]] = m
u = pd.read_csv(units).replace(np.inf, np.nan)
u = u[u.episode.isin(meta)]
e = u.groupby(["episode", "label"]).agg(lat=("ue_lat", "median"), rej=("ue_rej", "mean")).reset_index()
e["W"] = e.episode.map(lambda i: meta[i]["load_workers"])
e["cpu"] = e.episode.map(lambda i: meta[i]["vm_cpu_busy"])
e["alive"] = e.episode.map(lambda i: all(meta[i]["nf_alive_after"].values()))
rows = []
for W, g in e.groupby("W"):
    h, s = g[g.label == "healthy"], g[g.label != "healthy"]
    rows.append(dict(W=W, n=len(g), cpu=round(g.cpu.mean(), 2), healthy_med=round(h.lat.median(), 3),
                     healthy_q90=round(h.lat.quantile(0.9), 3), stall_med=round(s.lat.median(), 3),
                     all_alive=bool(g.alive.all()), max_rej=round(g.rej.max(), 2)))
r = pd.DataFrame(rows); print(r.to_string(index=False))
ok = r[(r.W > 0) & (r.healthy_q90 >= 0.87) & r.all_alive & (r.max_rej <= 0.5) & (r.stall_med > r.healthy_med)]
print("DECISION:", f"GO with W = {int(ok.W.min())}" if len(ok) else "NO-GO (load does not reproduce the disturbance)")
