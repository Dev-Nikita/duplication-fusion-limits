"""Phase A -> frozen originals. Reads the raw probe records of one physical run, applies the thresholds fixed before
the pilot (config/thresholds.json) and writes
  <run>/reports.jsonl : one whole report per firing source and episode (pattern (L, E) != (0, 0)), input of Phase B;
  <run>/episodes.csv  : per episode the state n = (n01, n10, n11) and the number of missing probe records.
No label is read here. Usage: python3 make_reports.py <run_dir> <run_id> <thresholds.json>"""
import sys, json, glob, os, csv


def main(run, run_id, thr_path):
    thr = json.load(open(thr_path))
    rec = {}
    for p in sorted(glob.glob(os.path.join(run, "phaseA", "probe_*.jsonl"))):
        for line in open(p):
            r = json.loads(line); rec[(r["episode"], r["source"])] = r
    eps = sorted({e for e, _ in rec}); srcs = sorted({s for _, s in rec})
    out = os.path.join(run, "reports.jsonl")
    if os.path.exists(out):
        sys.exit(f"{out} exists: originals are frozen")
    rows = []
    with open(out, "x") as f:
        for e in eps:
            n = {(0, 1): 0, (1, 0): 0, (1, 1): 0}; missing = 0
            for s in srcs:
                r = rec.get((e, s))
                if r is None or r["n_req"] < thr["min_requests"]:
                    missing += 1; continue
                L = int(r["p90_ms"] > thr["latency_p90_ms"])
                E = int((r["n_5xx"] + r["n_reset"] + r["n_timeout"]) / r["n_req"] >= thr["error_share"])
                if L or E:
                    n[(L, E)] += 1
                    f.write(json.dumps({"episode_id": f"{run_id}-{e}", "source_id": s, "root_event_id": f"{run_id}-{e}-{s}",
                                        "pattern": [L, E]}) + "\n")
            # state order of exact_core.patterns(2): (0,1), (1,0), (1,1) with coordinate 0 = L, coordinate 1 = E
            rows.append(dict(run=run_id, episode=e, episode_id=f"{run_id}-{e}", n01=n[(0, 1)], n10=n[(1, 0)], n11=n[(1, 1)],
                             missing=missing, M=len(srcs)))
    with open(os.path.join(run, "episodes.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{run_id}: {len(eps)} episodes, {sum(1 for _ in open(out))} positive reports")


if __name__ == "__main__":
    main(*sys.argv[1:4])
