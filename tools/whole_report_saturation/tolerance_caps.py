"""Exact vs certified tolerance caps for whole-report copying (Prop. 8): for each law of the 315-law study and
eps in {0.001, 0.005, 0.01, 0.02}, D_eps^exact = min{D: G_D <= eps}, D_eps^cert = min{D: B_D <= eps}, B_D the bound.
Reads out/quant_gap_cfg*.csv (exact rationals). Usage (from results/whole_report): python3 tolerance_caps.py"""
import csv, glob
from fractions import Fraction as F
from collections import defaultdict
rows = [r for f in sorted(glob.glob("out/quant_gap_cfg*.csv")) for r in csv.DictReader(open(f))]
laws = defaultdict(list)
for r in rows:
    laws[(r["K"], r["M"], r["kind"], r["trial"])].append(r)
out = []
for eps in ("1/1000", "1/200", "1/100", "1/50"):
    e = F(eps); diffs = []
    for key, rs in laws.items():
        rs = sorted(rs, key=lambda r: int(r["D"]))
        de = next(int(r["D"]) for r in rs if F(r["G"]) <= e); dc = next(int(r["D"]) for r in rs if F(r["UB"]) <= e)
        assert de <= dc
        diffs.append(dc - de)
    out.append(dict(eps=float(e), laws=len(diffs), equal=sum(d == 0 for d in diffs), larger_by_1=sum(d == 1 for d in diffs),
                    larger_by_2plus=sum(d >= 2 for d in diffs), max_diff=max(diffs)))
with open("out/tolerance_caps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for o in out:
    print(o)
