"""R7 feasibility pilot (2026-10-19): do natural repeated events exist that could stand in for copies?
Counts, per episode, UEs with more than one registration attempt, NAS timer/retransmission lines in UE logs,
AMF 'epoll failed' lines (an artefact of SIGSTOP injection) and repeated AMF log templates.
Usage: python3 -I pilot_natural_repeats.py <out_dir_with_episodes> [blocks...]"""
import collections, glob, json, os, re, sys
import pandas as pd
root = sys.argv[1]; blocks = {int(b) for b in sys.argv[2:]} if len(sys.argv) > 2 else None
rows = []
for ep in sorted(glob.glob(os.path.join(root, "episodes", "ep_*"))):
    m = json.load(open(os.path.join(ep, "meta.json"))); b = m.get("block")
    if blocks is not None and b not in blocks:
        continue
    att, tim = collections.Counter(), 0
    for f in ("ue_prov.log", "ue_unprov.log"):
        p = os.path.join(ep, f)
        if not os.path.exists(p):
            continue
        for l in open(p, errors="ignore"):
            mm = re.match(r"^\[[^\]]*\] \[(\d+)\|nas\] \[\w+\] (.*)$", l.strip())
            if not mm:
                continue
            if "Sending Initial Registration" in mm.group(2):
                att[mm.group(1)] += 1
            if re.search("T35|expir|retrans", mm.group(2), re.I):
                tim += 1
    amf = open(os.path.join(ep, "logs", "amf.log"), errors="ignore").read()
    rows.append(dict(label=m["label"], block=b, ues=len(att), retry_ues=sum(v > 1 for v in att.values()),
                     nas_timer_lines=tim, epoll_lines=amf.count("epoll failed")))
d = pd.DataFrame(rows)
print(len(d), "episodes"); print(d.groupby("label")[["ues", "retry_ues", "nas_timer_lines", "epoll_lines"]].agg(["mean", "max"]).round(2))
