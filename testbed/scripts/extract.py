#!/usr/bin/env python3
"""Post-process episodes INSIDE the core container (needs tshark):
  capture.pcap -> signalling.csv (NGAP procedures, 5GMM NAS message types, SBI HTTP/2 method/path/status)
  NF logs      -> logs/<nf>.log (byte slice of the episode)
Idempotent; skips episodes already extracted. Usage: python3 /scripts/extract.py
"""
import glob, json, os, subprocess

LOGDIR = "/opt/open5gs/var/log/open5gs"
FIELDS = ["frame.number", "frame.time_epoch", "ip.src", "ip.dst", "sctp.srcport", "tcp.srcport", "tcp.dstport",
          "ngap.procedureCode", "nas_5gs.mm.message_type", "http2.streamid", "http2.headers.method",
          "http2.headers.path", "http2.headers.status"]

for ep in sorted(glob.glob("/out/episodes/ep_*")):
    meta_p = os.path.join(ep, "meta.json")
    out_csv = os.path.join(ep, "signalling.csv")
    if not os.path.exists(meta_p) or os.path.exists(out_csv):
        continue
    meta = json.load(open(meta_p))
    cmd = ["tshark", "-r", os.path.join(ep, "capture.pcap"), "-d", "tcp.port==7777,http2",
           "-Y", "ngap || http2.header", "-T", "fields", "-E", "separator=;", "-E", "header=y", "-E", "occurrence=a"]
    for f in FIELDS:
        cmd += ["-e", f]
    with open(out_csv + ".tmp", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, check=False)
    os.replace(out_csv + ".tmp", out_csv)
    os.makedirs(os.path.join(ep, "logs"), exist_ok=True)
    for name, start in meta.get("log_offsets_start", {}).items():
        end = meta.get("log_offsets_end", {}).get(name, start)
        with open(os.path.join(LOGDIR, name), "rb") as src, open(os.path.join(ep, "logs", name), "wb") as dst:
            src.seek(start); dst.write(src.read(max(0, end - start)))
    print("extracted", os.path.basename(ep), meta["label"])
print("EXTRACT DONE")
