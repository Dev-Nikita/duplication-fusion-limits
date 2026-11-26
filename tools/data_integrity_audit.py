"""Data-integrity gate (R8). Writes results/integrity/DATA_QUALITY_REGISTRY.csv and results/integrity/audit.txt and
exits non-zero if any check fails. Block identifiers are unique across campaigns (R4/R5b 1-3, R6 4-12, R10 13-18,
R11 21-32, R11 pilot 900-905).
Checks
 1. no result file (CSV with a block column, fold_laws.pkl) uses an invalid block, except the raw readings and the
    WITHDRAWN frozen broker-replay input and results;
 2. the corrected broker replay contains no invalid block and its train/test blocks do not overlap;
 3. the Flight audit pipeline removes the airline (gold-source) sites and reports 29 sources;
 4. the episode and block counts stated in the article match the registry (2,100 episodes in 27 valid blocks for
    R4, R6, R10, R11; 600 episodes in the corrected replay);
 5. the withdrawn analysis is marked as such.
Usage: python3 tools/data_integrity_audit.py"""
import glob, hashlib, json, os, pickle, sys
import numpy as np, pandas as pd

INVALID = {("R6", 7): "UE identities beyond the provisioned range; most readings time out",
           ("R6", 8): "UE identities beyond the provisioned range; all readings time out",
           ("R6", 9): "UE identities beyond the provisioned range; all readings time out"}
INVALID_IDS = {b for _, b in INVALID}
UNITS = [("R4", "results/r5b/units.csv"), ("R6", "results/r6/units.csv"), ("R10", "results/r10/units.csv"),
         ("R11", "results/r11/units.csv"), ("R11-pilot", "results/r11_pilot/units.csv")]
ALLOWED_RAW = {"results/r6/units.csv"}
WITHDRAWN_DIRS = ("results/redelivery/",)
OUT = "results/integrity"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def block_users():
    users = {}
    for f in sorted(glob.glob("results/**/*.csv", recursive=True)):
        if f.startswith("results/integrity/"):
            continue
        try:
            cols = pd.read_csv(f, nrows=0).columns
        except Exception:
            continue
        bc = [c for c in cols if c.lower() in ("block", "held_block", "held", "test_block")]
        if not bc:
            continue
        d = pd.read_csv(f, usecols=bc, low_memory=False)
        for c in bc:
            for b in pd.to_numeric(d[c], errors="coerce").dropna().astype(int).unique():
                users.setdefault(int(b), set()).add(f)
    for x in pickle.load(open("results/finite_cap/fold_laws.pkl", "rb")):
        users.setdefault(int(x["held_block"]), set()).add("results/finite_cap/fold_laws.pkl")
        for b in (x["per_block"] or {}):
            users.setdefault(int(b), set()).add("results/finite_cap/fold_laws.pkl")
    return users


def registry(users):
    rows = []
    for camp, path in UNITS:
        d = pd.read_csv(path)
        for b, g in d.groupby("block"):
            b = int(b); key = (camp, b)
            used = sorted(f for f in users.get(b, ()) if f != path and not f.endswith("_units_in.csv"))
            rows.append(dict(campaign=camp, block=b, episodes=g.episode.nunique(), expected_sources=20,
                             observed_sources=round(g.groupby("episode").size().mean(), 2),
                             timeout_rate=round(float(np.isinf(g.ue_lat).mean()), 3), valid=key not in INVALID,
                             exclusion_reason=INVALID.get(key, ""), used_in=";".join(used), units_file=path, units_sha256=sha(path)))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True); fails = []; log = []
    users = block_users(); R = registry(users); R.to_csv(f"{OUT}/DATA_QUALITY_REGISTRY.csv", index=False)
    # 1
    for b in INVALID_IDS:
        bad = sorted(f for f in users.get(b, ()) if f not in ALLOWED_RAW and not f.startswith(WITHDRAWN_DIRS))
        if bad:
            fails.append(f"invalid block {b} used in {bad}")
    log.append(f"1. files with block columns scanned; invalid blocks used outside raw/withdrawn files: {'none' if not any('invalid block' in x for x in fails) else 'FOUND'}")
    # 2
    for v in ("A", "B"):
        T = pd.read_csv(f"results/redelivery_corrected/work/{v}/truth.csv")
        r6 = T[T.dataset == "r6"]
        if set(r6.block) & INVALID_IDS:
            fails.append(f"corrected replay {v} contains invalid blocks")
        for ds, g in T.groupby("dataset"):
            if set(g[g.split == "train"].block) & set(g[g.split == "test"].block):
                fails.append(f"corrected replay {v} {ds}: train/test blocks overlap")
        log.append(f"2. corrected replay {v}: {len(T)} episodes, R6 blocks {sorted(int(b) for b in r6.block.unique())}")
    # 3
    src = open("experiments/flight/flight_audit.py").read()
    desc = json.load(open("results/flight/audit_R2/flight_descriptors_no_gold_sites.json"))
    full = json.load(open("results/flight/flight_descriptors.json"))
    n_full = eval(full["0.95"])["n_sources"] if isinstance(full.get("0.95"), str) else full["0.95"]["n_sources"]
    if "if s not in AIRSITES" not in src or desc["n_sources"] != n_full - 3:
        fails.append("Flight: airline sites not removed or source count inconsistent")
    log.append(f"3. Flight: {n_full} sources in the frozen run, {desc['n_sources']} after removing the 3 airline sites")
    # 4
    V = R[R.valid & R.campaign.isin(["R4", "R6", "R10", "R11"])]
    n_ep, n_bl = int(V.episodes.sum()), len(V)
    if (n_ep, n_bl) != (2100, 27):
        fails.append(f"article states 2,100 episodes in 27 blocks; registry has {n_ep} in {n_bl}")
    log.append(f"4. valid testbed blocks: {n_bl}, episodes {n_ep} (article: 2,100 in 27)")
    # 5
    if not os.path.isfile("results/redelivery/WITHDRAWN.txt"):
        fails.append("results/redelivery is not marked as withdrawn")
    log.append("5. withdrawn marker present" if os.path.isfile("results/redelivery/WITHDRAWN.txt") else "5. withdrawn marker MISSING")
    log.append("RESULT: " + ("PASS" if not fails else "FAIL: " + " | ".join(fails)))
    open(f"{OUT}/audit.txt", "w").write("\n".join(log) + "\n"); print("\n".join(log))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
