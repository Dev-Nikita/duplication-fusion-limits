"""Window features from RCAEval logs (development on RE3, confirmation on RE2; docs/r12-protocol.md).
Per case three 180-s windows relative to the injection time t0:
  H (history, normal)  [t0-600, t0-420]   -- reference available to a deployed detector
  N (test, normal)     [t0-360, t0-180]   -- label 0
  F (test, fault)      [t0+60,  t0+240]   -- label 1
Raw quantities per window (system-agnostic, no service names):
  A   error lines                         (count channel: every message counted, copies included)
  U   distinct (pod, template) error units (lineage channel: copies of the same message from the same pod removed)
  P   distinct pods with an error line
  T   distinct error templates
  L   all log lines                       (the non-error lines L - A are the natural non-firing channel)
  K   distinct pods with any log line
Source id: pod_name if the file has it, else container_name (amendment A1, before any RE2 feature was computed).
Error line = message matching ERR (case-insensitive). Template = message with uuids, hex ids, numbers and
quoted strings masked. Usage: python3 rca_features.py <dataset dir> <out.csv> [system filter substring]"""
import glob, os, re, sys
import pandas as pd

ERR = re.compile(r"error|exception|fail|refused|timeout|timed out|unavailable|denied|\bwarn|cannot|could not|unable", re.I)
MASK = [(re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<uuid>"),
        (re.compile(r"\b[0-9a-f]{16,}\b", re.I), "<hex>"), (re.compile(r"\d+(\.\d+)?"), "<n>"),
        (re.compile(r"'[^']*'|\"[^\"]*\""), "<s>")]
WINDOWS = {"H": (-600, -420), "N": (-360, -180), "F": (60, 240)}


def template(m):
    m = str(m)[:300]
    for rx, rep in MASK:
        m = rx.sub(rep, m)
    return m


def main():
    root, out = sys.argv[1], sys.argv[2]; filt = sys.argv[3] if len(sys.argv) > 3 else ""
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "**", "logs.csv"), recursive=True)):
        case = os.path.relpath(os.path.dirname(f), root)
        if filt not in case:
            continue
        try:
            t0 = int(open(os.path.join(os.path.dirname(f), "inject_time.txt")).read().strip()[:10])
            cols = pd.read_csv(f, nrows=0).columns
            src = "pod_name" if "pod_name" in cols else "container_name"   # amendment A1 (RE2 has no pod_name)
            d = pd.read_csv(f, usecols=["timestamp", "message", src], dtype=str).rename(columns={src: "pod_name"})
        except Exception as e:  # noqa: BLE001
            print("skip", case, e, flush=True); continue
        t = pd.to_numeric(d.timestamp, errors="coerce") / 1e9
        parts = case.split(os.sep)
        for w, (a, b) in WINDOWS.items():
            x = d[(t >= t0 + a) & (t < t0 + b)]
            err = x.message.astype(str).str.contains(ERR)
            e = x[err]
            tpl = e.message.map(template) if len(e) else pd.Series(dtype=str)
            rows.append(dict(case=case, system=parts[-3] if len(parts) >= 3 else parts[0], fault_dir=parts[-2],
                             rep=parts[-1], window=w, A=int(err.sum()),
                             U=int(pd.DataFrame({"p": e.pod_name.values, "t": tpl.values}).drop_duplicates().shape[0]) if len(e) else 0,
                             P=int(e.pod_name.nunique()), T=int(tpl.nunique()) if len(e) else 0,
                             L=int(len(x)), K=int(x.pod_name.nunique())))
        print(case, flush=True)
    pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)


if __name__ == "__main__":
    main()
