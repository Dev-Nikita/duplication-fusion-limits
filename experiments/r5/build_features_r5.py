"""R5 features: the frozen R4 extractors (experiments/r4/build_features.py, sha256 ad9a7151...) applied
unchanged to R5 episodes (meta.block in {1,2,3}). Episodes without meta.json (interrupted) are listed and
excluded; if a schedule position was run twice (interruption between meta and state write) the first is kept.
Usage: python3 -I build_features_r5.py <testbed/out> <features.csv>"""
import glob, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r4"))
from build_features import amf_log, metrics, signalling, ue_logs  # noqa: E402

root, outp = sys.argv[1], sys.argv[2]
rows, skipped = [], []
for ep in sorted(glob.glob(os.path.join(root, "episodes", "ep_*"))):
    idx = int(ep.rsplit("_", 1)[1]); mp = os.path.join(ep, "meta.json")
    if not os.path.exists(mp):
        skipped.append(idx); continue
    meta = json.load(open(mp))
    if meta.get("block") not in (1, 2, 3):
        continue
    r = {"episode": idx, "label": meta["label"], "intensity": meta["intensity"], "block": meta["block"],
         "sched_pos": meta.get("sched_pos"), "netem_ok": meta.get("netem_ok"), "pilot": False}
    for fn in (signalling, amf_log, ue_logs, metrics):
        try:
            r.update(fn(ep))
        except Exception as e:
            r[f"err_{fn.__name__}"] = str(e)[:120]
    rows.append(r)
df = pd.DataFrame(rows)
n0 = len(df); dup = df.sched_pos.notna() & df.duplicated(["block", "sched_pos"], keep="first"); df = df[~dup]
bad_net = df[(df.label == "net_delay") & (df.netem_ok != True)]  # noqa: E712
df = df.drop(bad_net.index)
df.to_csv(outp, index=False)
print(f"{len(df)} R5 episodes -> {outp}; interrupted (no meta): {skipped}; duplicates dropped: {n0 - len(df) - len(bad_net)}; "
      f"net_delay without netem: {len(bad_net)}")
print(df.groupby(["block", "label"]).size().unstack(fill_value=0))
