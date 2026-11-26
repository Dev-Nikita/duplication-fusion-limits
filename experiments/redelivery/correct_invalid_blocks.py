"""Correction (24 Nov 2026, post hoc, after the frozen run): R6 blocks 7-9 are invalid testbed blocks (IMSIs beyond
the provisioned range; 78-100% of UE readings time out). The R6 analysis of the paper excludes them
(experiments/r6/run_r6.py, BLOCKS), but prepare.py used every block of results/r6/units.csv, so these 150 episodes
entered the TRAINING part of the broker replay and the p90 firing threshold. The held-out blocks (R6: 10-12;
R10: 17-18) are valid and unchanged; R10 is unaffected.
Two corrected inputs are written from the recorded trace (no new broker run):
  A  drop R6 blocks 7-9 (truth and deliveries); firing threshold as recorded (0.4891 s)
  B  as A, and re-threshold R6 at the p90 of finite healthy latencies of valid training blocks 4-6; deliveries of
     readings at or below the new threshold are removed (latency = severity * old threshold, severity from input/reports.jsonl; severity is capped at 10,
     which lies above any threshold ratio here).
Usage: python3 correct_invalid_blocks.py confirm/out input/truth.csv ../../results/r6/units.csv <work dir>"""
import json, os, sys
import numpy as np, pandas as pd
OUT, TRUTH, UNITS, WORK = sys.argv[1:5]
INVALID = {7, 8, 9}
T = pd.read_csv(TRUTH)
bad = set(T[(T.dataset == "r6") & T.block.isin(INVALID)].episode_id)
u = pd.read_csv(UNITS); h = u[u.block.isin([4, 5, 6]) & (u.label == "healthy")].ue_lat
thr_old = float(T[T.dataset == "r6"].threshold.iloc[0]); thr_new = float(np.percentile(h[np.isfinite(h)], 90))
sev = pd.read_json(os.path.join(os.path.dirname(TRUTH), "reports.jsonl"), lines=True).set_index("root_event_id").severity
for var in ("A", "B"):
    d_out = os.path.join(WORK, var, "out"); os.makedirs(d_out, exist_ok=True)
    keep_root = None
    for reg in ("S0", "S1", "S2", "S3a", "S3b", "S4"):
        Dl = pd.read_json(os.path.join(OUT, f"deliveries_{reg}.jsonl"), lines=True)
        Dl = Dl[~Dl.episode_id.isin(bad)]
        if var == "B":
            r6 = Dl.episode_id.str.startswith("r6-")
            Dl = Dl[~r6 | (Dl.root_event_id.map(sev) * thr_old > thr_new)]
        Dl.to_json(os.path.join(d_out, f"deliveries_{reg}.jsonl"), orient="records", lines=True)
        if reg == "S0":
            U = Dl.groupby("episode_id").source_id.nunique()
    Tv = T[~T.episode_id.isin(bad)].copy()
    if var == "B":
        Tv["U"] = Tv.episode_id.map(U).fillna(0).astype(int)
        Tv.loc[Tv.dataset == "r6", "threshold"] = thr_new
    Tv.to_csv(os.path.join(WORK, var, "truth.csv"), index=False)
    print(var, "episodes", len(Tv), Tv.groupby(["dataset", "split"]).size().to_dict(), "thr_r6", Tv[Tv.dataset == "r6"].threshold.iloc[0])
print("threshold old/new", thr_old, thr_new)
