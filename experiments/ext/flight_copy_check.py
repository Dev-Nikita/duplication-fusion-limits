"""Feasibility check (2026-10-24): does the real Flight dataset (Li et al., VLDB 2012; lunadong.com) show
evidence multiplicity, i.e. copying that distorts count-based fusion? Attribute: actual departure time.
Values are normalised to minutes of day; a claim equals the truth within 'tol' minutes.
Fusers: majority vote over sources (counts copies), and majority vote over agreement clusters (one vote per
cluster; a cluster = sources whose pairwise agreement on co-reported items is >= thr, single-linkage).
Clusters are built from ALL items without using the truth. Accuracy is measured on the gold items.
Usage: python3 -I flight_copy_check.py <clean_flight_dir> <flight_truth_dir>"""
import collections, glob, itertools, os, re, sys
import numpy as np

TIME = re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp])?")


def norm(s):
    m = TIME.search(s or "")
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
    if ap == "p" and h < 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    return h * 60 + mi


def load(data_dir, truth_dir):
    days = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*-data.txt"))):
        day = os.path.basename(f)[:10]; tf = os.path.join(truth_dir, f"{day}-truth.txt")
        if not os.path.exists(tf):
            continue
        claims = collections.defaultdict(dict)
        for line in open(f, encoding="utf-8", errors="ignore"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                v = norm(p[3])
                if v is not None:
                    claims[p[1]][p[0]] = v
        truth = {}
        for line in open(tf, encoding="utf-8", errors="ignore"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and norm(p[2]) is not None:
                truth[p[0]] = norm(p[2])
        days.append((day, claims, truth))
    return days


def eq(a, b, tol):
    d = abs(a - b) % 1440
    return min(d, 1440 - d) <= tol


def vote(cands, tol):
    """cands: list of (value, weight). Returns the value with the largest tolerant support."""
    best, bw = None, -1
    for v, _ in cands:
        w = sum(w2 for v2, w2 in cands if eq(v, v2, tol))
        if w > bw:
            best, bw = v, w
    return best


def main():
    days = load(sys.argv[1], sys.argv[2]); tol = 1
    srcs = sorted({s for _, c, _ in days for it in c.values() for s in it})
    agree, co = collections.Counter(), collections.Counter()
    for _, claims, _ in days:
        for it in claims.values():
            for a, b in itertools.combinations(sorted(it), 2):
                co[(a, b)] += 1; agree[(a, b)] += eq(it[a], it[b], tol)
    print(f"days {len(days)}, sources {len(srcs)}, gold items {sum(len(t) for *_, t in days)}")
    for thr in (0.9, 0.95, 0.99):
        parent = {s: s for s in srcs}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for (a, b), n in co.items():
            if n >= 50 and agree[(a, b)] / n >= thr:
                parent[find(a)] = find(b)
        cl = collections.defaultdict(list)
        for s in srcs:
            cl[find(s)].append(s)
        sizes = sorted((len(v) for v in cl.values()), reverse=True)
        acc = {"sources": [], "clusters": []}
        for _, claims, truth in days:
            for item, t in truth.items():
                it = claims.get(item)
                if not it:
                    continue
                acc["sources"].append(eq(vote([(v, 1.0) for v in it.values()], tol), t, tol))
                per = collections.defaultdict(list)
                for s, v in it.items():
                    per[find(s)].append(v)
                cand = [(v, 1.0 / len(vs)) for vs in per.values() for v in vs]
                acc["clusters"].append(eq(vote(cand, tol), t, tol))
        n = len(acc["sources"])
        print(f"thr {thr}: clusters {len(cl)} (largest {sizes[:6]}), gold items scored {n}, "
              f"vote over sources {np.mean(acc['sources']):.3f}, one vote per cluster {np.mean(acc['clusters']):.3f}")
        big = [sorted(v) for v in cl.values() if len(v) >= 3]
        for g in big[:4]:
            print("   cluster:", ", ".join(g[:12]) + (" ..." if len(g) > 12 else ""))


if __name__ == "__main__":
    main()
