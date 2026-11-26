"""Phase-B audit with the standard library only (runs on the server after each replay).
Checks per regime: every original delivered at least once; multiplicity <= D+1 (S6 is the stress regime above the
cap and is reported, not failed); pattern identical on every copy; payload fields limited to the whitelist (no label,
no injector field). Usage: python3 audit_phaseb.py <run_dir> [D]   (exit 1 if not clean)"""
import sys, json, glob, os

ALLOWED = {"episode_id", "source_id", "root_event_id", "pattern"}


def main(run, D=2):
    rep = {}
    for l in open(os.path.join(run, "reports.jsonl")):
        r = json.loads(l); rep[r["root_event_id"]] = r
    keys = set().union(*[set(r) for r in rep.values()]) if rep else set()
    ok = not (keys - ALLOWED)
    print(f"payload fields: {sorted(keys)}; outside whitelist: {sorted(keys - ALLOWED)}")
    for p in sorted(glob.glob(os.path.join(run, "phaseB", "deliveries_*.jsonl"))):
        rg = p.rsplit("_", 1)[1][:-6]; mult = {}; bad = 0
        for l in open(p):
            d = json.loads(l); mult[d["root_event_id"]] = max(mult.get(d["root_event_id"], 0), d["num_delivered"])
            if list(rep[d["root_event_id"]]["pattern"]) != list(d["pattern"]):
                bad += 1
        never = sum(1 for k in rep if mult.get(k, 0) == 0); over = sum(1 for v in mult.values() if v > D + 1)
        hist = {}
        for v in mult.values():
            hist[v] = hist.get(v, 0) + 1
        print(f"{rg}: originals {len(rep)}, never delivered {never}, above cap {over}, pattern changed {bad}, "
              f"deliveries per original {dict(sorted(hist.items()))}")
        if never or bad or (over and rg != "S6"):
            ok = False
    print("AUDIT", "CLEAN" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2) else 1)
