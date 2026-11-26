"""Audit and evaluation of a redelivery run (protocol: PROTOCOL_FROZEN.md).
Usage: python3 -W ignore evaluate.py <out dir with deliveries_*.jsonl> <truth.csv> <tools dir> <results dir> [--pilot]
--pilot: the truth file has training blocks only; the first block of each campaign is used to fit and the second to
exercise the code. Pilot numbers are an engineering check and are not reported as results."""
import collections, glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, sys.argv[3])
import finite_cap_cvx as C, frontier as FR  # noqa: E402

OUT, RES, PILOT = sys.argv[1], sys.argv[4], "--pilot" in sys.argv
D_DECLARED = 2; EPS = 0.2; TRAIN_REGIME = "S1"; CAP = {"S0": 3, "S1": 3, "S2": 3, "S3a": 3, "S3b": 5, "S4": 3}
os.makedirs(RES, exist_ok=True); T = pd.read_csv(sys.argv[2]).set_index("episode_id"); lines = []
def say(s=""):
    print(s); lines.append(s)

# ---------------- audit of the lineage (auditor side) ----------------
A = {}; audit_rows = []
for path in sorted(glob.glob(os.path.join(OUT, "deliveries_*.jsonl"))):
    reg = os.path.basename(path)[11:-6]; Dl = pd.read_json(path, lines=True); g = Dl.groupby("root_event_id")
    mult = g.size(); bad = dict(
        unknown_episode=int((~Dl.episode_id.isin(T.index)).sum()),
        two_sequences=int((g.stream_sequence.nunique() > 1).sum()),
        counters_not_1_to_n=int(sum(sorted(v) != list(range(1, len(v) + 1)) for v in g.num_delivered.apply(list))),
        cap_exceeded=int((mult > CAP[reg]).sum()),
        root_not_episode_source=int((Dl.root_event_id != Dl.episode_id + "-" + Dl.source_id.astype(str)).sum()))
    U_obs = Dl.groupby("episode_id").source_id.nunique().reindex(T.index).fillna(0).astype(int)
    bad["originals_missing"] = int((U_obs != T.U).sum())
    A[reg] = Dl.groupby("episode_id").size().reindex(T.index).fillna(0).astype(int)
    audit_rows.append(dict(regime=reg, deliveries=len(Dl), roots=len(mult), copies=int(len(Dl) - len(mult)), max_multiplicity=int(mult.max()),
                           share_roots_copied=float((mult > 1).mean()), bytes_per_delivery=float(Dl.bytes.mean()), **bad))
AU = pd.DataFrame(audit_rows); AU.to_csv(os.path.join(RES, "audit.csv"), index=False)
say("AUDIT"); say(AU.to_string(index=False))
viol = int(AU[["unknown_episode", "two_sequences", "counters_not_1_to_n", "cap_exceeded", "root_not_episode_source", "originals_missing"]].values.sum())
say(f"audit violations: {viol}")
quality_ok = viol == 0 and all(AU.set_index("regime").copies[r] > 0 for r in AU.regime if r != "S0") and AU.set_index("regime").copies.get("S0", 0) == 0
say(f"quality gate (no violations; copies in every regime but S0; none in S0): {'PASS' if quality_ok else 'FAIL'}")

# ---------------- rules ----------------
def law(y, v, m):
    p = np.zeros((2, m + 1))
    for a, b in zip(y, v):
        p[int(a), min(int(b), m)] += 1
    return p / p.sum()

def table(y, v, m):
    n = np.zeros((2, m + 1))
    for a, b in zip(y, v):
        n[int(a), min(int(b), m)] += 1
    eta = pd.Series(np.where(n.sum(0) > 0, (n[1] + 0.5) / (n.sum(0) + 1), np.nan)).ffill().bfill().values
    return lambda q: eta[np.minimum(np.asarray(q, int), m)]

rows = []
for ds, Td in T.groupby("dataset"):
    if PILOT:
        b = sorted(Td.block.unique()); tr = Td.block == b[0]; te = Td.block == b[-1]
    else:
        tr = Td.split == "train"; te = Td.split == "test"
    y = Td.y.values; U = Td.U.values; M = int(Td.M.max()); n = (D_DECLARED + 1) * M + 1; a1 = A[TRAIN_REGIME].loc[Td.index].values
    auc = roc_auc_score(y[tr], U[tr]); say(f"{ds}: learnability gate on training blocks, AUROC of U = {auc:.3f} ({'PASS' if auc >= 0.65 else 'FAIL'}; threshold 0.65)")
    p = law(y[tr], U[tr], M); q = law(y[tr], a1[tr], n - 1); pte = law(y[te], U[te], M)
    xm = C.ld_minimax(p, D_DECLARED)[1]; xw = FR.restricted(q, p, D_DECLARED, EPS)
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1, random_state=0).fit(a1[tr].reshape(-1, 1), y[tr])
    R = {"presence": table(y[tr], (U[tr] > 0).astype(int), 1), "count_table": table(y[tr], a1[tr], n - 1),
         "count_gbdt": lambda v: gb.predict_proba(np.asarray(v).reshape(-1, 1))[:, 1],
         "clipped_count": (lambda f: (lambda v: f(np.minimum(v, M))))(table(y[tr], np.minimum(a1[tr], M), M)),
         "log_count": (lambda f: (lambda v: f(np.floor(2 * np.log2(1 + np.asarray(v))).astype(int))))(table(y[tr], np.floor(2 * np.log2(1 + a1[tr])).astype(int), 16)),
         "minimax": lambda v: xm[np.minimum(np.asarray(v, int), n - 1)], "weighted": lambda v: xw[np.minimum(np.asarray(v, int), n - 1)]}
    oracle = table(y[tr], U[tr], M)
    for reg in sorted(A):
        a = A[reg].loc[Td.index].values[te]
        for name, f in R.items():
            pr = f((a > 0).astype(int)) if name == "presence" else f(a)
            rows.append(dict(dataset=ds, regime=reg, rule=name, brier=float(np.mean((pr - y[te]) ** 2)), n=int(te.sum())))
        rows.append(dict(dataset=ds, regime=reg, rule="oracle_unique (reference)", brier=float(np.mean((oracle(U[te]) - y[te]) ** 2)), n=int(te.sum())))
    for name, x in (("minimax", xm), ("weighted", xw)):
        rows.append(dict(dataset=ds, regime="model worst case (D=2)", rule=name, brier=FR.worst(x, pte, D_DECLARED), n=int(te.sum())))
R_ = pd.DataFrame(rows); R_.to_csv(os.path.join(RES, "brier.csv"), index=False)
piv = R_.pivot_table(index=["dataset", "rule"], columns="regime", values="brier").round(4)
say(); say("HELD-OUT BRIER (observed, by regime) and model worst case" + (" -- PILOT CODE CHECK, NOT A RESULT" if PILOT else "")); say(piv.to_string())
if not PILOT:
    obs = R_[~R_.regime.str.startswith("model")]; within = ["S1", "S2", "S3a", "S4"]; say(); say("PRE-SPECIFIED READINGS (margin 0.010 Brier)")
    for ds, g in obs.groupby("dataset"):
        w = g[g.regime.isin(within)].pivot(index="rule", columns="regime", values="brier"); worst = w.max(1); s1 = w["S1"]
        h1 = min(worst["minimax"], worst["weighted"]) <= worst["count_table"] - 0.010
        h2 = s1["count_table"] <= s1["minimax"] + 0.010
        say(f"{ds}: worst observed over S1,S2,S3a,S4: count_table {worst['count_table']:.4f}, count_gbdt {worst['count_gbdt']:.4f}, minimax {worst['minimax']:.4f}, weighted {worst['weighted']:.4f}, presence {worst['presence']:.4f}")
        say(f"{ds}: H1 (a robust rule beats the trained count table on the worst observed regime by at least 0.010): {'HELD' if h1 else 'NOT HELD'}")
        say(f"{ds}: H2 (under stable delivery S1 the count table is not worse than minimax by more than 0.010): {'HELD' if h2 else 'NOT HELD'}")
open(os.path.join(RES, "summary.txt"), "w").write("\n".join(lines) + "\n")
