"""Frozen evaluation for R12 (docs/r12-protocol.md). Usage: python3 -I rca_evaluate.py <cells.csv> <out.json>
Cluster bootstrap (B = 5000, seed 20261024) over fault groups (fault_dir = service x fault type, replicates as one
cluster), resampled within each held-out system. Primary learner: gbdt; logit reported without a test.
Gate: a system is analysed only if within-system AUROC (gbdt) of COUNT or UNIT is >= 0.65.

PRIMARY (worst case over held-out systems, the criterion of R10 P1 and R11-X):
  H-W : max_s Brier_cross(COUNT, s) - max_s Brier_cross(UNIT, s) > 0  -> bootstrap CI95 lower bound > 0
SECONDARY (tested with the same rule, reported in full):
  H-WB: max_s Brier_cross(BOTH, s)  - max_s Brier_cross(UNIT, s) > 0
  H-A : mean case Brier_cross(COUNT) - Brier_cross(UNIT) > 0 (pooled), CI95 lower bound > 0
  H-C : worst-case gap (COUNT - UNIT) larger in cross than in within: [H-W gap]_cross - [same]_within > 0
Decision: H-W confirmed -> GO (the natural-duplication result enters the paper as confirmatory);
otherwise R12 is reported as it is and the paper remains a boundary study."""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

B, SEED = 5000, 20261024


def auroc(g):
    y = np.r_[np.zeros(len(g)), np.ones(len(g))]; p = np.r_[g.p_N.values, g.p_F.values]
    ok = ~np.isnan(p)
    return float(roc_auc_score(y[ok], p[ok])) if len(set(y[ok])) == 2 else np.nan


def stat(piv, scheme, a, b, kind):
    x = piv[piv.scheme == scheme]
    if kind == "worst":
        return x.groupby("test_system")[a].mean().max() - x.groupby("test_system")[b].mean().max()
    return (x[a] - x[b]).mean()


def boot(piv, schemes, a, b, kind, rng):
    """Resample fault groups within each held-out system (the same draw for every scheme), using per-group
    sums so each replicate is cheap. Mean over cases is preserved (weights = cases per group)."""
    systems = sorted(piv.test_system.unique())
    agg = piv.groupby(["scheme", "test_system", "fault_dir"]).agg(sa=(a, "sum"), sb=(b, "sum"), n=(a, "size")).reset_index()
    fds = {s: sorted(agg[agg.test_system == s].fault_dir.unique()) for s in systems}
    tab = {(sc, s): agg[(agg.scheme == sc) & (agg.test_system == s)].set_index("fault_dir").reindex(fds[s]).fillna(0)[["sa", "sb", "n"]].values
           for sc in schemes for s in systems}
    out = np.empty(B)
    for i in range(B):
        draw = {s: rng.integers(0, len(fds[s]), len(fds[s])) for s in systems}
        vals = []
        for sc in schemes:
            ma, mb, tot = [], [], np.zeros(3)
            for s in systems:
                t = tab[(sc, s)][draw[s]].sum(0)
                ma.append(t[0] / t[2]); mb.append(t[1] / t[2]); tot += t
            vals.append(max(ma) - max(mb) if kind == "worst" else (tot[0] - tot[1]) / tot[2])
        out[i] = vals[0] if len(vals) == 1 else vals[0] - vals[1]
    return out


def test(piv, schemes, a, b, kind):
    pt = [stat(piv, sc, a, b, kind) for sc in schemes]
    est = pt[0] if len(pt) == 1 else pt[0] - pt[1]
    bs = boot(piv, schemes, a, b, kind, np.random.default_rng(SEED))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    per = piv[piv.scheme == schemes[0]].groupby("test_system")[[a, b]].mean().round(4).to_dict("index")
    return dict(est=round(float(est), 5), ci95=[round(float(lo), 5), round(float(hi), 5)], per_system_brier=per,
                confirmed=bool(lo > 0))


def main():
    c = pd.read_csv(sys.argv[1]); res = {"auroc": {}, "brier": {}}
    for (sch, ln, fs, s), g in c.groupby(["scheme", "learner", "set", "test_system"]):
        res["auroc"][f"{sch}|{ln}|{fs}|{s}"] = round(auroc(g), 4); res["brier"][f"{sch}|{ln}|{fs}|{s}"] = round(g.brier.mean(), 5)
    gw = c[(c.scheme == "within") & (c.learner == "gbdt")]
    keep = [s for s in sorted(c.test_system.unique())
            if max(auroc(gw[(gw.test_system == s) & (gw["set"] == fs)]) for fs in ("COUNT", "UNIT")) >= 0.65]
    res["gate_systems"] = keep
    if not keep:
        # The empty analysis set is a pre-declared terminal outcome, not an
        # exceptional state. Do not pivot it into missing feature columns.
        res["status"] = "NOT_TESTABLE"
        res["decision"] = {
            "testable": False,
            "H_W": None,
            "H_WB": None,
            "H_A": None,
            "H_C": None,
            "GO": False,
        }
        with open(sys.argv[2], "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: res[k] for k in ("status", "gate_systems", "decision")}, indent=1))
        return
    for ln in ("gbdt", "logit"):
        x = c[(c.learner == ln) & (c.test_system.isin(keep))]
        piv = x.pivot_table(index=["scheme", "test_system", "fault_dir", "case"], columns="set", values="brier").reset_index()
        res[ln] = {"H-W": test(piv, ["cross"], "COUNT", "UNIT", "worst"),
                   "H-WB": test(piv, ["cross"], "BOTH", "UNIT", "worst"),
                   "H-A": test(piv, ["cross"], "COUNT", "UNIT", "mean"),
                   "H-C": test(piv, ["cross", "within"], "COUNT", "UNIT", "worst"),
                   "RATE_vs_UNIT_worst": test(piv, ["cross"], "RATE", "UNIT", "worst")}
    g = res["gbdt"]
    res["status"] = "TESTED"
    res["decision"] = dict(testable=True, H_W=g["H-W"]["confirmed"], H_WB=g["H-WB"]["confirmed"], H_A=g["H-A"]["confirmed"],
                           H_C=g["H-C"]["confirmed"], GO=bool(g["H-W"]["confirmed"]))
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("gate_systems", "decision")}, indent=1))
    for ln in ("gbdt", "logit"):
        for k, v in res[ln].items():
            print(ln, k, v["est"], v["ci95"], v["confirmed"], v["per_system_brier"])


if __name__ == "__main__":
    main()
