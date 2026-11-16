"""External study on naturally copied evidence: the Flight data set (protocol docs/flight-protocol.md).
Binary task per gold item (day, flight): y = departure delayed by >= 15 min (gold actual vs gold schedule).
A source fires if its reported actual departure is >= 15 min after the published (gold) schedule; it sends a
negative report if it reports an actual departure that is earlier than that; otherwise it is silent.
Copy clusters (lineage proxy) are built on TRAINING items only, without the truth: single linkage over source
pairs with >= 50 co-reported items and agreement (within 1 min) >= tau.
Channels: A (firing sources), U (firing clusters), P = 1{A>0}, N (negative-reporting sources),
NU (clusters that report and do not fire).
Usage: python3 -I experiments/flight/flight_study.py <clean_dir> <truth_dir> <out_dir> [--permute-labels SEED]"""
import collections, itertools, json, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import flight_io as F  # noqa: E402
import finite_cap_cvx as C, finite_cap_solvers as S, saturation_closed_form as SC, quantitative_saturation as QS  # noqa: E402

THR_DELAY = 15; TAUS = (0.95, 0.90, 0.99); MIN_CO = 50; TOL = 1; SEED = 20261115
GB = dict(max_depth=3, max_iter=200, learning_rate=0.1, random_state=SEED)


def clusters(items, tau):
    """items: list of dict source -> claimed actual departure (all items of the training days)."""
    agree, co = collections.Counter(), collections.Counter(); srcs = set()
    for it in items:
        srcs.update(it)
        for a, b in itertools.combinations(sorted(it), 2):
            co[(a, b)] += 1
            d = abs(it[a] - it[b]) % 1440; agree[(a, b)] += min(d, 1440 - d) <= TOL
    parent = {s: s for s in srcs}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (a, b), n in co.items():
        if n >= MIN_CO and agree[(a, b)] / n >= tau:
            parent[find(a)] = find(b)
    return {s: find(s) for s in srcs}


def features(rows, cl):
    out = []
    for r in rows:
        fire = {s for s, v in r["claims"].items() if F.delay(v, r["sched"]) >= THR_DELAY}
        rep = set(r["claims"]); fc = {cl.get(s, s) for s in fire}; rc = {cl.get(s, s) for s in rep}
        out.append(dict(A=len(fire), U=len(fc), N=len(rep) - len(fire), NU=len(rc - fc), P=int(len(fire) > 0)))
    return pd.DataFrame(out)


def gbdt(Xtr, ytr, Xte):
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(np.mean(ytr)))
    return HistGradientBoostingClassifier(**GB).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def law(y, u, m):
    p = np.zeros((2, m + 1))
    for a, b in zip(y, u):
        p[int(a), min(int(b), m)] += 1
    return p / p.sum()


def table_rule(p):
    s = p.sum(0); eta = np.where(s > 0, p[1] / np.where(s > 0, s, 1), np.nan)
    return pd.Series(eta).ffill().bfill().fillna(0.5).values


def run_split(rows, y, groups, all_items_by_day, tau, name):
    """Leave-one-group-out. Returns per-item predictions for every rule."""
    preds = []
    for g in sorted(set(groups)):
        te = np.array([x == g for x in groups]); tr = ~te
        tr_days = {rows[i]["day"] for i in np.where(tr)[0]}
        if name == "airline":       # clusters from the training airlines' items on all days
            items = [it for d in all_items_by_day for fl, it in all_items_by_day[d].items() if fl.split("-")[0] != g]
        else:                        # clusters from the training days' items
            items = [it for d in tr_days for it in all_items_by_day[d].values()]
        cl = clusters(items, tau); sizes = collections.Counter(cl.values()); D = max(sizes.values()) - 1; mc = len(sizes)
        Ftr = features([rows[i] for i in np.where(tr)[0]], cl); Fte = features([rows[i] for i in np.where(te)[0]], cl)
        ytr = y[tr]; res = pd.DataFrame(dict(idx=np.where(te)[0], y=y[te], group=g, split=name, tau=tau, D_hat=D, n_clusters=mc))
        for rule, cols in (("count", ["A"]), ("unique", ["U"]), ("presence", ["P"]), ("count_neg", ["A", "N"]), ("unique_neg", ["U", "NU"])):
            res["p_" + rule] = gbdt(Ftr[cols].values, ytr, Fte[cols].values)
        p = law(ytr, Ftr.U.values, mc); eta = table_rule(p)
        res["p_naive"] = eta[np.minimum(Fte.A.values, mc)]
        x = C.ld_minimax(p, D)[1]
        res["p_minimax"] = x[np.minimum(Fte.A.values, (D + 1) * mc)]
        for c in ("A", "U", "N", "NU", "P"):
            res[c] = Fte[c].values
        preds.append(res)
    return pd.concat(preds, ignore_index=True)


def main():
    data = F.load(sys.argv[1], sys.argv[2]); out = sys.argv[3]; os.makedirs(out, exist_ok=True)
    rows, all_items = [], {}
    for day, claims, truth in data:
        all_items[day] = claims
        for fl, (sched, actual) in sorted(truth.items()):
            rows.append(dict(day=day, flight=fl, airline=fl.split("-")[0], sched=sched, claims=claims.get(fl, {}),
                             y=int(F.delay(actual, sched) >= THR_DELAY)))
    y = np.array([r["y"] for r in rows])
    if "--permute-labels" in sys.argv:
        y = np.random.default_rng(int(sys.argv[sys.argv.index("--permute-labels") + 1])).permutation(y)
    days = sorted(all_items); week = {d: min(i // 7, 3) for i, d in enumerate(days)}   # 4 date blocks (last block: days 22-31)
    allp = []
    for tau in TAUS:
        allp.append(run_split(rows, y, [week[r["day"]] for r in rows], all_items, tau, "week"))
        allp.append(run_split(rows, y, [r["airline"] for r in rows], all_items, tau, "airline"))
    P = pd.concat(allp, ignore_index=True); P["day"] = [rows[i]["day"] for i in P.idx]
    P.to_csv(os.path.join(out, "flight_predictions.csv"), index=False)
    # in-sample descriptors on all items, clusters from all items (primary tau)
    desc = {}
    for tau in TAUS:
        cl = clusters([it for d in all_items for it in all_items[d].values()], tau); sizes = collections.Counter(cl.values())
        D = max(sizes.values()) - 1; mc = len(sizes); Fa = features(rows, cl)
        p = law(y, Fa.U.values, mc); pA = law(y, Fa.A.values, int(Fa.A.max()))
        rbu, rbp, rba = S.bayes_risk_U(p), S.bayes_risk_presence(p), S.bayes_risk_U(pA)
        rstar = C.ld_minimax(p, D)[0]; rli = C.li_minimax(p, D)[0]; _, h = QS.constants(p)
        fire = Fa[Fa.U > 0]; mult = (fire.A / fire.U)
        desc[str(tau)] = dict(n_items=len(rows), n_sources=len(cl), n_clusters=mc, cluster_sizes=sorted(sizes.values(), reverse=True), D_hat=D,
                              RB_U=rbu, RB_P=rbp, RB_A_natural=rba, R_star_Dhat=rstar, R_countonly_Dhat=rli,
                              nature_index=(rba - rbu) / (rstar - rbu) if rstar > rbu else float("nan"),
                              D_sat=int(np.ceil(SC.rho(p) - 1e-12) - 1), h=h, delta_Dhat=QS.delta(p, D), delta_1=QS.delta(p, 1), delta_2=QS.delta(p, 2),
                              copies_per_firing_cluster_y0=float(mult[y[fire.index] == 0].mean()), copies_per_firing_cluster_y1=float(mult[y[fire.index] == 1].mean()),
                              max_A_over_U=float(mult.max()))
    json.dump(desc, open(os.path.join(out, "flight_descriptors.json"), "w"), indent=1)
    print("done", len(P))


if __name__ == "__main__":
    main()
