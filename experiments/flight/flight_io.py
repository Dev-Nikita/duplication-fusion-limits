"""Loader for the Flight data set (Li et al., VLDB 2012; lunadong.com/fusionDataSets.htm).
Line format: source \t flight \t sched_dep \t actual_dep \t dep_gate \t sched_arr \t actual_arr \t arr_gate.
Truth format: flight \t sched_dep \t actual_dep \t dep_gate \t sched_arr \t actual_arr \t arr_gate.
Times are reduced to minutes of day (local). delay(a, s) = signed difference in (-720, 720]."""
import collections, glob, os, re

TIME = re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp])?")


def tod(s):
    m = TIME.search(s or "")
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
    if h > 23 or mi > 59:
        return None
    if ap == "p" and h < 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    return h * 60 + mi


def delay(actual, sched):
    d = (actual - sched) % 1440
    return d - 1440 if d > 720 else d


def load(data_dir, truth_dir):
    """Returns list of (day, claims, truth): claims[flight][source] = actual-departure minute of day;
    truth[flight] = (sched_dep, actual_dep) minutes of day (both present)."""
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*-data.txt"))):
        day = os.path.basename(f)[:10]; tf = os.path.join(truth_dir, f"{day}-truth.txt")
        if not os.path.exists(tf):
            continue
        claims = collections.defaultdict(dict)
        for line in open(f, encoding="utf-8", errors="ignore"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and tod(p[3]) is not None:
                claims[p[1]][p[0]] = tod(p[3])
        truth = {}
        for line in open(tf, encoding="utf-8", errors="ignore"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and tod(p[1]) is not None and tod(p[2]) is not None:
                truth[p[0]] = (tod(p[1]), tod(p[2]))
        out.append((day, claims, truth))
    return out
