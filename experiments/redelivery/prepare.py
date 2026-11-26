"""Builds the producer input and the auditor file from the per-unit testbed readings.
  reports.jsonl : one line per firing reading (episode_id, source_id, root_event_id, severity); no label, block or split
  truth.csv     : auditor only: episode_id, dataset, block, split, y, U, M
A reading fires if its UE-side latency exceeds the 90th percentile of the finite healthy readings in the TRAINING
blocks of its campaign; a reading that timed out (infinite latency) fires. Split by physical block: the last third of the blocks of each campaign is held out. y = 1 for a fault episode.
Usage: python3 prepare.py <r6 units.csv> <r10 units.csv> <out dir> [--pilot]   (--pilot: first two training blocks only)"""
import json, math, os, sys
import numpy as np, pandas as pd

out = sys.argv[3]; os.makedirs(out, exist_ok=True); pilot = "--pilot" in sys.argv; rep = []; truth = []
for ds, path in (("r6", sys.argv[1]), ("r10", sys.argv[2])):
    d = pd.read_csv(path); blocks = sorted(d.block.unique()); k = math.ceil(len(blocks) / 3); test = set(blocks[-k:]); train = [b for b in blocks if b not in test]
    h = d[d.block.isin(train) & (d.label == "healthy")].ue_lat; thr = float(np.percentile(h[np.isfinite(h)], 90))
    if pilot:
        d = d[d.block.isin(train[:2])]
    for ep, g in d.groupby("episode", sort=True):
        eid = f"{ds}-{ep}"; fire = g[g.ue_lat > thr]
        truth.append(dict(episode_id=eid, dataset=ds, block=int(g.block.iloc[0]), split="test" if g.block.iloc[0] in test else "train",
                          y=int(g.label.iloc[0] != "healthy"), U=len(fire), M=len(g), threshold=thr))
        for _, r in fire.iterrows():
            rep.append(dict(episode_id=eid, source_id=str(r.imsi), root_event_id=f"{eid}-{r.imsi}", severity=round(min(float(r.ue_lat) / thr, 10.0), 4)))
with open(os.path.join(out, "reports.jsonl"), "w") as f:
    for r in rep:
        f.write(json.dumps(r) + "\n")
pd.DataFrame(truth).to_csv(os.path.join(out, "truth.csv"), index=False)
print(len(rep), "reports;", len(truth), "episodes;", pd.DataFrame(truth).groupby(["dataset", "split"]).size().to_dict())
