"""Feasibility pilot on RCAEval RE3 logs (2026-10-24). Counts only; no outcome model is fitted.
Questions: (1) are there natural repeated error messages ("copies")? (2) does their multiplicity depend on the
state (normal vs fault), i.e. natural label-dependent duplication? (3) is there a unit to define distinct
sources (pod, template)? (4) do the same pods emit non-error lines (a natural "non-firing" channel)?
Windows per case: normal = [inject-540 s, inject-240 s], fault = [inject+60 s, inject+360 s].
Usage: python3 rcaeval_pilot.py <RE3 dir> <out.csv>"""
import glob, os, re, sys
import pandas as pd

ERR = re.compile(r"error|exception|fail|refused|timeout|timed out|unavailable|denied|\bwarn|cannot|could not|unable", re.I)
MASK = [(re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<uuid>"),
        (re.compile(r"\b[0-9a-f]{16,}\b", re.I), "<hex>"), (re.compile(r"\d+(\.\d+)?"), "<n>"),
        (re.compile(r"'[^']*'|\"[^\"]*\""), "<s>")]


def template(m):
    m = str(m)[:300]
    for rx, rep in MASK:
        m = rx.sub(rep, m)
    return m


rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*", "*", "logs.csv"))):
    case = os.path.relpath(os.path.dirname(f), sys.argv[1])
    try:
        inj = int(open(os.path.join(os.path.dirname(f), "inject_time.txt")).read().strip()[:10])
        d = pd.read_csv(f, usecols=["timestamp", "container_name", "message", "pod_name"], dtype=str)
    except Exception as e:  # noqa: BLE001
        print("skip", case, e); continue
    t = pd.to_numeric(d.timestamp, errors="coerce") / 1e9
    for win, (a, b) in (("normal", (inj - 540, inj - 240)), ("fault", (inj + 60, inj + 360))):
        w = d[(t >= a) & (t < b)].copy()
        if w.empty:
            continue
        w["err"] = w.message.astype(str).str.contains(ERR)
        e = w[w.err].copy(); e["tpl"] = e.message.map(template)
        units = e.groupby(["pod_name", "tpl"]).size()
        rows.append(dict(system=case.split(os.sep)[0], case=case, fault=case.split(os.sep)[1], window=win,
                         lines=len(w), err_lines=len(e), err_units=len(units),
                         multiplicity=(len(e) / len(units)) if len(units) else 0.0,
                         max_mult=int(units.max()) if len(units) else 0,
                         err_pods=e.pod_name.nunique(), pods=w.pod_name.nunique(),
                         nonerr_lines=int((~w.err).sum())))
    print(case, flush=True)
out = pd.DataFrame(rows); out.to_csv(sys.argv[2], index=False)
s = out.groupby(["system", "window"])[["lines", "err_lines", "err_units", "multiplicity", "max_mult", "err_pods", "nonerr_lines"]].median()
print(s.round(2).to_string())
p = out.pivot_table(index="case", columns="window", values=["err_lines", "err_units", "multiplicity"])
for k in ("err_lines", "err_units", "multiplicity"):
    if ("fault" in p[k]) and ("normal" in p[k]):
        r = (p[k]["fault"] + 1) / (p[k]["normal"] + 1)
        print(f"{k}: fault/normal ratio median {r.median():.2f}, share of cases with ratio > 1.5: {(r > 1.5).mean():.2f}")
