"""Flight addendum R2 (docs/flight-addendum-R2.md): gold-source overlap audit, exclusion modes and same-channel baselines.
Post-discovery sensitivity analysis; the frozen study files are imported unchanged.
Usage: python3 -I -W ignore experiments/flight/flight_audit.py <clean_dir> <truth_dir> <out_dir>"""
import collections, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flight_io as F, flight_study as ST  # noqa: E402

SITE = {"AA": "aa", "UA": "ua", "CO": "CO"}; AIRSITES = set(SITE.values()); TAU = 0.95; rng = np.random.default_rng(20261116)


def raw_sched(data_dir):
    """claims' scheduled departure per (day, flight, source), for the schedule check."""
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*-data.txt"))):
        day = os.path.basename(f)[:10]
        for line in open(f, encoding="utf-8", errors="ignore"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and F.tod(p[2]) is not None:
                out[(day, p[1], p[0])] = F.tod(p[2])
    return out


def build(data, mode):
    rows, items = [], {}
    for day, claims, truth in data:
        cl = {}
        for fl, c in claims.items():
            al = fl.split("-")[0]
            if mode == "A":
                c = {s: v for s, v in c.items() if s != SITE.get(al)}
            elif mode == "B":
                c = {s: v for s, v in c.items() if s not in AIRSITES}
            cl[fl] = c
        items[day] = cl
        for fl, (sched, actual) in sorted(truth.items()):
            rows.append(dict(day=day, flight=fl, airline=fl.split("-")[0], sched=sched, claims=cl.get(fl, {}),
                             y=int(F.delay(actual, sched) >= ST.THR_DELAY)))
    return rows, items


def extra_rules(P, rows, y, groups_of):
    """Same-channel rules on the held-out predictions frame P (one split, primary tau): needs A, N of train and test."""
    out = P.copy(); A = np.zeros(len(rows)); N = np.zeros(len(rows))
    # features A, N do not depend on clusters
    for i, r in enumerate(rows):
        fire = sum(F.delay(v, r["sched"]) >= ST.THR_DELAY for v in r["claims"].values()); A[i] = fire; N[i] = len(r["claims"]) - fire
    frac = np.where(A + N > 0, A / np.maximum(A + N, 1), -1.0); groups = np.array(groups_of)
    for col in ("p_table_A", "p_frac", "p_coverage", "p_neg_only", "p_majority", "p_frac_table"):
        out[col] = np.nan
    for g in sorted(set(groups_of)):
        te = groups == g; tr = ~te; m = out.idx.isin(np.where(te)[0]).values; idx = out.idx[m].values
        mA = int(A.max()); eta = ST.table_rule(ST.law(y[tr], A[tr], mA)); out.loc[m, "p_table_A"] = eta[np.minimum(A[idx].astype(int), mA)]
        out.loc[m, "p_frac"] = ST.gbdt(frac[tr].reshape(-1, 1), y[tr], frac[idx].reshape(-1, 1))
        out.loc[m, "p_coverage"] = ST.gbdt((A + N)[tr].reshape(-1, 1), y[tr], (A + N)[idx].reshape(-1, 1))
        out.loc[m, "p_neg_only"] = ST.gbdt(N[tr].reshape(-1, 1), y[tr], N[idx].reshape(-1, 1))
        out.loc[m, "p_majority"] = (A[idx] > N[idx]).astype(float)
        b = lambda f: np.where(f < 0, 10, np.minimum((f * 10).astype(int), 9))
        bt, bi = b(frac[tr]), b(frac[idx]); tab = np.array([(y[tr][bt == k].sum() + 1) / ((bt == k).sum() + 2) for k in range(11)])
        out.loc[m, "p_frac_table"] = tab[bi]
    return out


def main():
    data = F.load(sys.argv[1], sys.argv[2]); out = sys.argv[3]; os.makedirs(out, exist_ok=True)
    # ---- 1. overlap audit
    sch = raw_sched(sys.argv[1]); ov = []; ag = collections.defaultdict(lambda: [0, 0]); sg = [0, 0]
    for day, claims, truth in data:
        for fl, (sched, actual) in truth.items():
            al = fl.split("-")[0]; c = claims.get(fl, {}); own = SITE.get(al)
            ov.append(dict(day=day, flight=fl, airline=al, n_sources=len(c), own_site_reports=int(own in c),
                           own_site_equals_gold=int(own in c and abs(F.delay(c[own], actual)) <= 1) if own in c else np.nan,
                           any_airline_site=int(any(s in AIRSITES for s in c))))
            for s, v in c.items():
                ag[(s, al)][1] += 1; ag[(s, al)][0] += abs(F.delay(v, actual)) <= 1
                if (day, fl, s) in sch:
                    sg[1] += 1; sg[0] += abs(F.delay(sch[(day, fl, s)], sched)) <= 1
    O = pd.DataFrame(ov); O.to_csv(os.path.join(out, "flight_gold_overlap.csv"), index=False)
    AG = pd.DataFrame([dict(source=s, airline=al, n=t, agree_gold=a / t) for (s, al), (a, t) in ag.items()]).sort_values(["airline", "agree_gold"], ascending=[True, False])
    AG.to_csv(os.path.join(out, "flight_source_gold_agreement.csv"), index=False)
    summ = dict(n_items=len(O), own_site_reports=int(O.own_site_reports.sum()), own_site_share=float(O.own_site_reports.mean()),
                own_site_equals_gold_share=float(O.own_site_equals_gold.mean()), by_airline=O.groupby("airline").agg(n=("flight", "size"), own=("own_site_reports", "mean"), eq=("own_site_equals_gold", "mean")).round(4).to_dict("index"),
                sources_sched_equals_gold_sched=sg[0] / sg[1])
    # ---- 2-4. exclusion modes with the unchanged pipeline + extra rules
    res = []
    for mode in ("C", "A", "B"):
        rows, items = build(data, mode); y = np.array([r["y"] for r in rows])
        days = sorted(items); week = {d: min(i // 7, 3) for i, d in enumerate(days)}
        for split, groups in (("week", [week[r["day"]] for r in rows]), ("airline", [r["airline"] for r in rows])):
            P = ST.run_split(rows, y, groups, items, TAU, split); P["day"] = [rows[i]["day"] for i in P.idx]
            P = extra_rules(P, rows, y, groups)
            if mode == "C":      # own airline site alone
                sig = np.array([(1 if F.delay(r["claims"][SITE[r["airline"]]], r["sched"]) >= ST.THR_DELAY else 0) if SITE.get(r["airline"]) in r["claims"] else -1 for r in rows]); g = np.array(groups); P["p_own_site"] = np.nan
                for gg in sorted(set(groups)):
                    te = g == gg; m = P.idx.isin(np.where(te)[0]).values
                    P.loc[m, "p_own_site"] = ST.gbdt(sig[~te].reshape(-1, 1), y[~te], sig[P.idx[m].values].reshape(-1, 1))
            P["mode"] = mode; P.to_csv(os.path.join(out, f"audit_predictions_{mode}_{split}.csv"), index=False)
            rules = [c[2:] for c in P.columns if c.startswith("p_")]; B = {r: (P["p_" + r] - P.y) ** 2 for r in rules}
            row = dict(mode=mode, split=split, n=len(P), D_hat=f"{P.D_hat.min()}-{P.D_hat.max()}", n_clusters=f"{P.n_clusters.min()}-{P.n_clusters.max()}")
            row.update({"brier_" + r: float(B[r].mean()) for r in rules})
            dl = sorted(P.day.unique()); byday = {d: P.index[P.day == d].values for d in dl}
            for a, b in (("count", "unique"), ("count_neg", "unique"), ("frac", "count_neg"), ("frac_table", "count_neg"), ("majority", "count_neg"), ("table_A", "count"), ("naive", "table_A")):
                d = (B[a] - B[b]); bs = [d.loc[np.concatenate([byday[x] for x in rng.choice(dl, size=len(dl), replace=True)])].mean() for _ in range(2000)]
                row[f"d_{a}_minus_{b}"] = float(d.mean()); row[f"lo90_{a}_minus_{b}"] = float(np.quantile(bs, 0.05)); row[f"hi90_{a}_minus_{b}"] = float(np.quantile(bs, 0.95))
            row["groups"] = json.dumps({str(k): {r: round(float(B[r][P.group == k].mean()), 4) for r in ("count", "unique", "count_neg", "frac", "presence")} | {"n": int((P.group == k).sum()), "base_rate": round(float(P.y[P.group == k].mean()), 3)} for k in sorted(P.group.unique())})
            res.append(row); print(mode, split, flush=True)
    R = pd.DataFrame(res); R.to_csv(os.path.join(out, "flight_gold_exclusion_results.csv"), index=False)
    json.dump(summ, open(os.path.join(out, "flight_gold_overlap_summary.json"), "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    print(json.dumps(summ, indent=1)); print(R[[c for c in R.columns if c != "groups"]].round(4).T.to_string())
    for _, r in R.iterrows():
        print(r["mode"], r["split"], r["groups"])


if __name__ == "__main__":
    main()
