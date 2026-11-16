"""R5b per-UE unit layer. Designed on R4 episodes (ep_0008-0127) only, frozen before any R5 metric was
computed. For every UE of an episode, two independent observation channels give a per-UE record:
  UE log  (UERANSIM): ue_lat  = t(Initial Registration is successful) - t(first Sending Initial Registration)
                      ue_rej  = 'Initial Registration failed' seen
  AMF log (Open5GS):  amf_lat = t([imsi] Registration complete) - t(first line naming the UE's SUCI)
                      amf_rej = '[suci] Registration reject' seen
Missing completion -> latency = +inf (counts as slow). Output: one row per (episode, UE).
Usage: python3 -I build_units.py <testbed/out> <units.csv>"""
import glob, json, os, re, sys
from datetime import datetime
import pandas as pd

UE_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] \[(\d{15})\|nas\] \[\w+\] (.*)$")
AMF_RE = re.compile(r"^(\d\d/\d\d \d\d:\d\d:\d\d\.\d+): \[\w+\] \w+: (.*)$")
SUCI_RE = re.compile(r"\[suci-0-(\d{3})-(\d{2,3})-\d{4}-\d-\d-(\d+)\]")
IMSI_RE = re.compile(r"\[imsi-(\d{15})\]")


def ue_records(ep):
    rec = {}
    for f in ("ue_prov.log", "ue_unprov.log"):
        p = os.path.join(ep, f)
        if not os.path.exists(p):
            continue
        for line in open(p, errors="ignore"):
            m = UE_RE.match(line.strip())
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp(); imsi, msg = m.group(2), m.group(3)
            r = rec.setdefault(imsi, {"t0": None, "t_ok": None, "rej": 0})
            if "Sending Initial Registration" in msg and r["t0"] is None:
                r["t0"] = t
            elif "Initial Registration is successful" in msg and r["t_ok"] is None:
                r["t_ok"] = t
            elif "Initial Registration failed" in msg:
                r["rej"] = 1
    return rec


def amf_records(ep):
    rec = {}
    p = os.path.join(ep, "logs", "amf.log")
    if not os.path.exists(p):
        return rec
    for line in open(p, errors="ignore"):
        m = AMF_RE.match(line.strip())
        if not m:
            continue
        t = datetime.strptime("2026/" + m.group(1), "%Y/%m/%d %H:%M:%S.%f").timestamp(); msg = m.group(2)
        s = SUCI_RE.search(msg)
        if s:
            imsi = s.group(1) + s.group(2) + s.group(3)
            r = rec.setdefault(imsi, {"t0": None, "t_ok": None, "rej": 0})
            if r["t0"] is None:
                r["t0"] = t
            if "Registration reject" in msg:
                r["rej"] = 1
        i = IMSI_RE.search(msg)
        if i and "Registration complete" in msg:
            r = rec.setdefault(i.group(1), {"t0": None, "t_ok": None, "rej": 0})
            if r["t_ok"] is None:
                r["t_ok"] = t
    return rec


def lat(r):
    if r is None or r["t0"] is None or r["t_ok"] is None:
        return float("inf")
    return r["t_ok"] - r["t0"]


def main():
    root, outp = sys.argv[1], sys.argv[2]
    rows = []
    for ep in sorted(glob.glob(os.path.join(root, "episodes", "ep_*"))):
        mp = os.path.join(ep, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp)); idx = int(ep.rsplit("_", 1)[1])
        ue, amf = ue_records(ep), amf_records(ep)
        for imsi in sorted(ue):
            if ue[imsi]["t0"] is None:
                continue
            a = amf.get(imsi)
            rows.append(dict(episode=idx, label=meta["label"], intensity=meta["intensity"], block=meta.get("block"),
                             sched_pos=meta.get("sched_pos"), netem_ok=meta.get("netem_ok"), pilot=idx <= 7, imsi=imsi,
                             ue_lat=lat(ue[imsi]), ue_rej=ue[imsi]["rej"], amf_lat=lat(a),
                             amf_rej=(a or {"rej": 0})["rej"], amf_seen=int(a is not None)))
    df = pd.DataFrame(rows); df.to_csv(outp, index=False)
    print(f"{len(df)} UE rows from {df.episode.nunique()} episodes -> {outp}")


if __name__ == "__main__":
    main()
