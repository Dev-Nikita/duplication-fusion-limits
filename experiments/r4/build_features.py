"""R4 observation layer: per-episode readings from four naturally heterogeneous sources.
Fixed on the 8 pilot episodes (ep_0000-0007) and frozen before the campaign episodes were read.

Origins (what is observed) and readings (which source observes it); higher value = worse:
  REG  registration procedure  : reg_sig_t90 (pcap NAS), reg_log_p90 (AMF log), reg_ue_p90 (UE log),
                                 reg_met_failfrac (AMF Prometheus counters)
  AMF  AMF availability        : amf_sig_react_p90 (pcap NGAP), amf_met_scrape_err (Prometheus scrape
                                 failures), amf_log_maxgap (AMF log)
  AUTH authentication path     : auth_sbi_ausf_p90, auth_sbi_udm_p90 (pcap SBI latency), auth_sbi_udr_p90
  REJ  rejected registrations  : rej_sig (NAS Registration Reject count), rej_log (AMF log rejects),
                                 rej_ue (UE-side failures), rej_sbi404 (SBI 404 count), rej_met (reginitfail delta)
Usage: python3 build_features.py <out_dir_with_episodes> <features.csv> [--only-pilot]
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOGT = re.compile(r"^(\d\d)/(\d\d) (\d\d):(\d\d):(\d\d\.\d+):")
UET = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] \[(\d{15})\|")


def tod(line):
    m = LOGT.match(line)
    if not m:
        return None
    return int(m.group(3)) * 3600 + int(m.group(4)) * 60 + float(m.group(5))


def p90(v):
    return float(np.percentile(v, 90)) if len(v) else np.nan


def signalling(ep):
    f = os.path.join(ep, "signalling.csv")
    d = pd.read_csv(f, sep=";", dtype=str)
    d["t"] = d["frame.time_epoch"].astype(float)
    out = {}
    nas = d.dropna(subset=["nas_5gs.mm.message_type"])
    ev = [(t, m) for t, ms in zip(nas.t, nas["nas_5gs.mm.message_type"]) for m in ms.split(",")]
    req = [t for t, m in ev if m == "0x41"]
    # Registration Accept is integrity-protected (not decoded); NGAP InitialContextSetup (code 14) carries it
    ng = d.dropna(subset=["ngap.procedureCode"]).sort_values("t")
    ics = sorted(t for t, cs in zip(ng.t, ng["ngap.procedureCode"]) if "14" in cs.split(","))
    out["reg_sig_t90"] = (np.percentile(ics, 90) - min(req)) if req and ics else np.nan
    # AMF reaction: each gNB->AMF NGAP frame to the next AMF->gNB NGAP frame
    ul = ng[ng["ip.dst"] == "127.0.0.5"].t.to_numpy(); dl = ng[ng["ip.src"] == "127.0.0.5"].t.to_numpy()
    react = [dl[dl > t][0] - t for t in ul if (dl > t).any()]
    out["amf_sig_react_p90"] = p90(react)
    out["rej_sig"] = sum(1 for _, m in ev if m == "0x44")
    # SBI request/response latency, single-stream frames only
    h = d.dropna(subset=["http2.streamid"])
    h = h[~h["http2.streamid"].str.contains(",")]
    pend, lat = {}, {}
    for r in h.itertuples(index=False):
        sid = getattr(r, "_9")
        meth, path, st = r._10, r._11, r._12
        src, dst, sp, dp = r._2, r._3, r._5, r._6
        if isinstance(meth, str) and isinstance(path, str):
            pend[(src, sp, dst, dp, sid)] = (r.t, path)
        elif isinstance(st, str):
            k = (dst, dp, src, sp, sid)
            if k in pend:
                t0, path = pend.pop(k)
                svc = path.split("/")[1] if path.startswith("/") else path
                lat.setdefault(svc, []).append(r.t - t0)
                if st.split(",")[0] == "404":
                    out["rej_sbi404"] = out.get("rej_sbi404", 0) + 1
    out["auth_sbi_ausf_p90"] = p90(lat.get("nausf-auth", []))
    out["auth_sbi_udm_p90"] = p90(lat.get("nudm-ueau", []))
    out["auth_sbi_udr_p90"] = p90(lat.get("nudr-dr", []))
    out.setdefault("rej_sbi404", 0)
    return out


def amf_log(ep):
    p = os.path.join(ep, "logs", "amf.log")
    lines = [ANSI.sub("", l) for l in open(p, errors="ignore")] if os.path.exists(p) else []
    ts = [t for t in (tod(l) for l in lines) if t is not None]
    first, done = {}, {}
    rej = 0
    for l in lines:
        t = tod(l)
        if t is None:
            continue
        m = re.search(r"\[suci-0-\d+-\d+-\d+-\d+-\d+-(\d{10})\] Unknown UE by SUCI", l)
        if m:
            first.setdefault(m.group(1), t)
        m = re.search(r"\[imsi-\d{5}(\d{10})\] Registration complete", l)
        if m:
            done[m.group(1)] = t
        if re.search(r"[Rr]egistration reject|Registration Reject", l):
            rej += 1
    dur = [done[k] - first[k] for k in done if k in first and done[k] >= first[k]]
    gaps = np.diff(sorted(ts)) if len(ts) > 1 else np.array([np.nan])
    return {"reg_log_p90": p90(dur), "amf_log_maxgap": float(np.nanmax(gaps)), "rej_log": rej}


def ue_logs(ep):
    first, ok, fails = {}, {}, 0
    for p in glob.glob(os.path.join(ep, "ue_*.log")):
        for l in open(p, errors="ignore"):
            l = ANSI.sub("", l)
            m = UET.match(l)
            if not m:
                continue
            t = pd.Timestamp(m.group(1)).value / 1e9
            imsi = m.group(2)
            first.setdefault(imsi, t)
            if "Initial Registration is successful" in l:
                ok[imsi] = t
            if re.search(r"[Rr]egistration (failed|reject)", l):
                fails += 1
    dur = [ok[k] - first[k] for k in ok]
    return {"reg_ue_p90": p90(dur), "rej_ue": fails}


def metrics(ep):
    recs = [json.loads(l) for l in open(os.path.join(ep, "metrics.jsonl"))]
    amf = [r for r in recs if r["nf"] == "amf"]
    err = np.mean(["error" in r for r in amf]) if amf else np.nan
    good = [r for r in amf if "lines" in r]

    def val(r, name):
        s = 0.0
        for l in r["lines"]:
            if l.startswith(name):
                try:
                    s += float(l.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        return s

    out = {"amf_met_scrape_err": float(err)}
    if len(good) >= 2:
        a, b = good[0], good[-1]
        dreq = val(b, "fivegs_amffunction_rm_reginitreq") - val(a, "fivegs_amffunction_rm_reginitreq")
        dsucc = val(b, "fivegs_amffunction_rm_reginitsucc") - val(a, "fivegs_amffunction_rm_reginitsucc")
        dfail = val(b, "fivegs_amffunction_rm_reginitfail") - val(a, "fivegs_amffunction_rm_reginitfail")
        out["reg_met_failfrac"] = float(1 - dsucc / dreq) if dreq > 0 else np.nan
        out["rej_met"] = float(dfail)
    else:
        out["reg_met_failfrac"] = np.nan; out["rej_met"] = np.nan
    return out


def main():
    root, outp = sys.argv[1], sys.argv[2]
    only_pilot = "--only-pilot" in sys.argv
    rows = []
    for ep in sorted(glob.glob(os.path.join(root, "episodes", "ep_*"))):
        idx = int(ep.rsplit("_", 1)[1])
        if only_pilot and idx > 7:
            continue
        meta = json.load(open(os.path.join(ep, "meta.json")))
        r = {"episode": idx, "label": meta["label"], "intensity": meta["intensity"], "pilot": idx <= 7}
        for fn in (signalling, amf_log, ue_logs, metrics):
            try:
                r.update(fn(ep))
            except Exception as e:  # recorded, never silently dropped
                r[f"err_{fn.__name__}"] = str(e)[:120]
        rows.append(r)
    pd.DataFrame(rows).to_csv(outp, index=False)
    print(f"{len(rows)} episodes -> {outp}")


if __name__ == "__main__":
    main()
