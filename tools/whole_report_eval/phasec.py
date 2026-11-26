"""Phase C: audit, pilot gate and fusion evaluation of the K=2 whole-report experiment.

Usage:
  python3 phasec.py audit  <run_dir> [D]                  -> prints the Phase-B audit, exit 1 if not clean
  python3 phasec.py gate   <out_dir> <run_dir> ...          -> pilot gate (pilot runs only)
  python3 phasec.py eval   <out_dir> <design.json>          -> main analysis (train/test runs fixed in design.json)
design.json: {"train": [...], "test": [...], "regimes": ["S0",...,"S6"], "train_regime": "S1", "D": 2,
              "grace_s": 2.0, "eps": 0.005, "margin": 0.005, "caps": [0, 1, 2, 4, 9]}
The label is read only from <run>/truth/truth.jsonl and joined by episode here; rules never see it at inference.
"""
import sys, os, json, csv, glob, math
from fractions import Fraction as F
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wr_k2 as W  # noqa: E402

ALLOWED_PAYLOAD = {"episode_id", "source_id", "root_event_id", "pattern"}
PAT = {(0, 1): 0, (1, 0): 1, (1, 1): 2}


def ts(s):
    return pd.Timestamp(s).timestamp()


def load_run(run):
    ep = pd.read_csv(os.path.join(run, "episodes.csv"))
    tr = pd.DataFrame([json.loads(l) for l in open(os.path.join(run, "truth", "truth.jsonl"))])
    if "condition" not in tr:
        tr["condition"] = "all"
    tr = tr[["episode", "y", "condition", "k"]].rename(columns={"k": "n_affected"})
    ep = ep.merge(tr, on="episode", how="left")
    rep = {}
    for l in open(os.path.join(run, "reports.jsonl")):
        r = json.loads(l); rep[r["root_event_id"]] = r
    return ep, rep


def deliveries(run, regime):
    p = os.path.join(run, "phaseB", f"deliveries_{regime}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def episode_obs(run, regime, D, grace_s):
    """per episode: observed marginal counts A, copied pattern counts t, conformity and its reasons."""
    ep, rep = load_run(run)
    dl = deliveries(run, regime)
    pub = {}
    for l in open(os.path.join(run, "phaseB", f"published_{regime}.jsonl")):
        r = json.loads(l); pub.setdefault(r["episode_id"], []).append(ts(r["published_at"]))
    # window of episode e closes at the last publication of an original of any episode <= e, plus the grace time;
    # defined also for episodes without positive reports (they inherit the close of the preceding episodes)
    close, last = {}, -math.inf
    for e in ep.sort_values("episode").episode_id:
        if e in pub:
            last = max(last, max(pub[e]))
        close[e] = last + grace_s
    per_root = {}
    rows = {e: dict(A0=0, A1=0, t01=0, t10=0, t11=0, late=0, bytes=0) for e in ep.episode_id}
    for d in dl:
        e = d["episode_id"]; rr = rows[e]; per_root[d["root_event_id"]] = max(per_root.get(d["root_event_id"], 0), d["num_delivered"])
        if ts(d["received_at"]) > close[e]:
            rr["late"] += 1; continue
        L, E = d["pattern"]; rr["A0"] += L; rr["A1"] += E; rr[["t01", "t10", "t11"][PAT[(L, E)]]] += 1; rr["bytes"] += d["bytes"]
    never, over = {}, {}
    for root, r in rep.items():
        k = per_root.get(root, 0); e = r["episode_id"]
        if k == 0:
            never[e] = never.get(e, 0) + 1
        if k > D + 1:
            over[e] = over.get(e, 0) + 1
    out = ep.copy()
    for c in ("A0", "A1", "t01", "t10", "t11", "late", "bytes"):
        out[c] = [rows[e][c] for e in out.episode_id]
    out["never_delivered"] = [never.get(e, 0) for e in out.episode_id]
    out["above_cap"] = [over.get(e, 0) for e in out.episode_id]
    out["conforming"] = (out.never_delivered == 0) & (out.above_cap == 0) & (out.late == 0) & (out.missing == 0)
    out["regime"] = regime
    return out


def audit(run, D=2):
    _, rep = load_run(run); ok = True; lines = []
    for p in sorted(glob.glob(os.path.join(run, "phaseB", "deliveries_*.jsonl"))):
        rg = p.rsplit("_", 1)[1][:-6]; mult = {}; bad_pat = 0
        for l in open(p):
            d = json.loads(l); root = d["root_event_id"]
            mult[root] = max(mult.get(root, 0), d["num_delivered"])
            if list(rep[root]["pattern"]) != list(d["pattern"]):
                bad_pat += 1
        payload_keys = set().union(*[set(json.loads(l).keys()) for l in open(os.path.join(run, "reports.jsonl"))])
        never = sum(1 for r in rep if mult.get(r, 0) == 0); over = sum(1 for v in mult.values() if v > D + 1)
        leak = payload_keys - ALLOWED_PAYLOAD
        line = f"{rg}: originals {len(rep)}, never delivered {never}, above cap {over}, pattern changed {bad_pat}, payload fields outside whitelist {sorted(leak)}"
        lines.append(line)
        if never or bad_pat or leak or (over and rg != "S6"):
            ok = False
    return ok, lines


# ------------------------------------------------------------------ helpers
def law_of(ep):
    e = ep[ep.missing == 0]
    cnt = {}
    for y, a, b, c in zip(e.y, e.n01, e.n10, e.n11):
        k = (int(y), (int(a), int(b), int(c))); cnt[k] = cnt.get(k, 0) + 1
    return W.law_from_counts(cnt)


def auroc(score, y):
    s1, s0 = score[y == 1], score[y == 0]
    if len(s1) == 0 or len(s0) == 0:
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in s1 for b in s0]))


def tv_joint_vs_product(e):
    """TV distance between the joint law of the per-probe (L, E) indicators and the product of its marginals."""
    tot = e.M.sum() - e.missing.sum()
    p = {(1, 1): e.n11.sum() / tot, (1, 0): e.n10.sum() / tot, (0, 1): e.n01.sum() / tot}
    p[(0, 0)] = 1 - sum(p.values()); pl = p[(1, 1)] + p[(1, 0)]; pe = p[(1, 1)] + p[(0, 1)]
    q = {(a, b): (pl if a else 1 - pl) * (pe if b else 1 - pe) for a in (0, 1) for b in (0, 1)}
    return 0.5 * sum(abs(p[k] - q[k]) for k in p)


def spearman_probe(e):
    """|Spearman| (= |Pearson| for binary) of the per-probe L and E indicators."""
    tot = e.M.sum() - e.missing.sum(); n11 = e.n11.sum(); pl = (n11 + e.n10.sum()) / tot; pe = (n11 + e.n01.sum()) / tot
    den = math.sqrt(pl * (1 - pl) * pe * (1 - pe))
    return abs((n11 / tot - pl * pe) / den) if den > 0 else float("nan")


# ------------------------------------------------------------------ gate (pilot runs only)
def gate(out, runs, D=2):
    os.makedirs(out, exist_ok=True)
    ep = pd.concat([load_run(r)[0] for r in runs]); ep = ep[ep.missing == 0]
    y = ep.y.values; U0 = (ep.n10 + ep.n11).values; U1 = (ep.n01 + ep.n11).values
    allep = pd.concat([load_run(r)[0] for r in runs])
    miss = {f"missing_record_share_y{c}": float((allep[allep.y == c].missing > 0).mean()) for c in (0, 1)}
    res = dict(episodes=len(ep), share_y1=float(y.mean()), **miss,
               missing_share_by_affected=allep.groupby("n_affected").missing.apply(lambda m: float((m > 0).mean())).to_dict(),
               episodes_by_condition=allep.condition.value_counts().to_dict(), auroc_L=auroc(U0, y), auroc_E=auroc(U1, y),
               abs_spearman_probe=spearman_probe(ep), tv_joint_product=tv_joint_vs_product(ep),
               tv_joint_product_y0=tv_joint_vs_product(ep[ep.y == 0]), tv_joint_product_y1=tv_joint_vs_product(ep[ep.y == 1]))
    law = law_of(ep)
    st = {}
    for (yy, n), w in law.items():
        st.setdefault(yy, {})[n] = w
    res["distinct_states_y0"], res["distinct_states_y1"] = len(st.get(0, {})), len(st.get(1, {}))
    counts = ep.groupby(["n01", "n10", "n11"]).size()
    res["share_singleton_states"] = float((counts == 1).sum() / len(counts))
    res["mass_states_seen_lt3"] = float(counts[counts < 3].sum() / len(ep))
    res["pilot_H1_stat"] = float(W.bayes(W.channel_law(law, W.presence).values()) - W.minimax(law, "whole", W.INF)[0])
    res["pilot_J2"] = float(W.minimax(law, "coord", D)[0] - W.minimax(law, "whole", D)[0])
    aud = [audit(r, D) for r in runs]
    res["audit_clean"] = all(a[0] for a in aud)
    conf = {}
    for rg in ("S0", "S1", "S2", "S3", "S4", "S5"):
        if all(os.path.exists(os.path.join(r, "phaseB", f"deliveries_{rg}.jsonl")) for r in runs):
            o = pd.concat([episode_obs(r, rg, D, 2.0) for r in runs]); conf[rg] = float(o.conforming.mean())
    res["conforming_share"] = conf
    res["GO"] = bool(min(res["share_y1"], 1 - res["share_y1"]) >= 0.25 and abs(res["auroc_L"] - 0.5) >= 0.1 and abs(res["auroc_E"] - 0.5) >= 0.1
                     and res["abs_spearman_probe"] < 0.8 and res["audit_clean"] and res["tv_joint_product"] >= 0.02
                     and all(v >= 0.9 for v in conf.values()))
    json.dump(res, open(os.path.join(out, "gate.json"), "x"), indent=1)
    for a in aud:
        print("\n".join(a[1]))
    print(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ main evaluation
def evaluate(out, design):
    os.makedirs(out, exist_ok=True); D = design["D"]; caps = design.get("caps", [0, 1, 2, 4, 9]); eps = design["eps"]
    M = int(load_run(design["train"][0])[0].M.iloc[0])
    obs = {rg: {r: episode_obs(r, rg, D if rg != "S6" else D, design["grace_s"]) for r in design["train"] + design["test"]}
           for rg in design["regimes"]}
    trn = pd.concat([load_run(r)[0] for r in design["train"]]); ptr = law_of(trn)
    tro = pd.concat([obs[design["train_regime"]][r] for r in design["train"]]); tro = tro[tro.conforming]
    tr_obs = [(int(y), int(a), int(b)) for y, a, b in zip(tro.y, tro.A0, tro.A1)]
    pres = W.PresenceRule(tr_obs)
    rules = {"coordinate_presence": pres, "count_table": W.CountTable(tr_obs, pres),
             "logistic_log": W.Logistic(tr_obs, "log", M), "logistic_clip": W.Logistic(tr_obs, "clip", M),
             "minimax_whole_D": W.MinimaxRule(ptr, "whole", D, pres), "minimax_coord_D": W.MinimaxRule(ptr, "coord", D, pres),
             "minimax_whole_inf": W.MinimaxRule(ptr, "whole", W.INF, pres)}
    trn_ok = trn[trn.missing == 0]; prior = float(trn_ok.y.mean()); ppc = {}
    for y, a, b, c in zip(trn_ok.y, trn_ok.n01, trn_ok.n10, trn_ok.n11):
        k = (a > 0, b > 0, c > 0); n0, n1 = ppc.get(k, (0, 0)); ppc[k] = (n0 + (y == 0), n1 + (y == 1))
    pp_rule = {k: (n1 + prior) / (n0 + n1 + 1) for k, (n0, n1) in ppc.items()}  # copy-invariant channel, trained
    hyp, model, observed, excl, chan, sens = [], [], [], [], [], []
    for r in design["test"]:
        ep = load_run(r)[0]; pte = law_of(ep); rid = os.path.basename(r.rstrip("/"))
        RBpres = float(W.bayes(W.channel_law(pte, W.presence).values())); Vinf = float(W.minimax(pte, "whole", W.INF)[0])
        h = dict(run=rid, n_episodes=int((ep.missing == 0).sum()), H1_stat=RBpres - Vinf)
        wc = {k: W.worst_case(f, pte, "whole", D, "label", M) for k, f in rules.items() if k != "minimax_whole_inf"}
        h.update({f"WC_label_{k}": v for k, v in wc.items()})
        h["H2_min_margin"] = min(wc[k] for k in ("count_table", "logistic_log", "logistic_clip")) - wc["minimax_whole_D"]
        h["J_oracle"] = float(W.minimax(pte, "coord", D)[0] - W.minimax(pte, "whole", D)[0])
        h["J_deploy"] = wc["minimax_coord_D"] - wc["minimax_whole_D"]
        e_ok = ep[ep.missing == 0]
        pp = np.array([pp_rule.get((a > 0, b > 0, c > 0), prior) for a, b, c in zip(e_ok.n01, e_ok.n10, e_ok.n11)])
        h["pattern_presence_deployed"] = float(np.mean((pp - e_ok.y.values) ** 2))  # = its worst case: copies do not change it
        h["H4_margin"] = wc["minimax_whole_D"] - h["pattern_presence_deployed"]
        h["H4_oracle_margin"] = float(W.minimax(pte, "whole", D)[0]
                                      - W.bayes(W.channel_law(pte, lambda n: tuple(int(c > 0) for c in n)).values()))
        for cond, sub in e_ok.groupby("condition"):  # structural sensitivity (descriptive): nested, mixed, opposing
            pc = law_of(sub)
            if len({yy for yy, _ in pc}) < 2:
                continue
            sens.append(dict(run=rid, condition=cond, episodes=len(sub),
                             H1_stat=float(W.bayes(W.channel_law(pc, W.presence).values()) - W.minimax(pc, "whole", W.INF)[0]),
                             J_oracle=float(W.minimax(pc, "coord", D)[0] - W.minimax(pc, "whole", D)[0]),
                             H4_oracle_margin=float(W.minimax(pc, "whole", D)[0]
                                                    - W.bayes(W.channel_law(pc, lambda n: tuple(int(c > 0) for c in n)).values()))))
        dec = []
        for Dd in (1, 2, 4):
            Gt, Bt, _, _ = W.gap_bound(ptr, Dd); Ge, Be, _, _ = W.gap_bound(pte, Dd)
            dec.append(((Bt <= eps) == (Be <= eps)) and abs(float(Gt - Ge)) <= 0.01)
            h[f"G_train_{Dd}"], h[f"G_test_{Dd}"], h[f"B_train_{Dd}"], h[f"B_test_{Dd}"] = map(float, (Gt, Ge, Bt, Be))
        h["H3b_stable"] = all(dec)
        hyp.append(h)
        for k, f in rules.items():
            if k == "minimax_whole_inf":
                continue
            for Dd in caps:
                model.append(dict(run=rid, rule=k, D=Dd, wc_label=W.worst_case(f, pte, "whole", Dd, "label", M),
                                  wc_state=W.worst_case(f, pte, "whole", Dd, "state", M)))
        for rg in design["regimes"]:
            o = obs[rg][r]
            for k, f in rules.items():
                p = f(o.A0.values, o.A1.values); err = (p - o.y.values) ** 2
                observed.append(dict(run=rid, regime=rg, rule=k, brier_conforming=float(err[o.conforming.values].mean()),
                                     brier_all=float(err.mean())))
            for yy in (0, 1):
                s = o[o.y == yy]
                excl.append(dict(run=rid, regime=rg, y=yy, episodes=len(s), excluded_share=float(1 - s.conforming.mean()),
                                 missing_record=int((s.missing > 0).sum()),
                                 never=int(s.never_delivered.sum()), above_cap=int(s.above_cap.sum()), late=int(s.late.sum())))
        bytes_ep = float(obs[design["train_regime"]][r].bytes.mean())
        for ch, risk, bits, need in W.channel_table(pte, D, M):
            # aggregate_bits: size of the aggregated observation handed to the fuser (theoretical, no headers);
            # transport_bytes: measured broker payload per episode under the training regime. All messages still
            # reach the broker, so a smaller aggregate does not by itself reduce inbound traffic.
            chan.append(dict(run=rid, channel=ch, oracle_robust_risk=risk, aggregate_bits=bits, needs=need,
                             transport_bytes_per_episode=bytes_ep))
    H = pd.DataFrame(hyp); m = design["margin"]
    ex0 = pd.DataFrame(excl)
    summ = dict(H1=bool(H.H1_stat.mean() >= m and (H.H1_stat > 0).sum() >= math.ceil(0.75 * len(H))), H2=bool((H.H2_min_margin >= m).all()),
                H3a=bool(H.J_oracle.mean() >= m), H3b=bool(H.H3b_stable.all()), H4=bool(H.H4_margin.mean() >= m),
                H1_mean=H.H1_stat.mean(), J_oracle_mean=H.J_oracle.mean(), J_deploy_mean=H.J_deploy.mean())
    ex = pd.DataFrame(excl); gapcls = ex.groupby(["regime", "y"]).excluded_share.mean().unstack()
    summ["exclusion_flag_regimes"] = [rg for rg in gapcls.index if rg != "S6" and abs(gapcls.loc[rg, 0] - gapcls.loc[rg, 1]) > 0.05]
    share = 1 - ex0.groupby("regime").apply(lambda d: (d.excluded_share * d.episodes).sum() / d.episodes.sum())
    summ["low_conformity_regimes"] = [rg for rg, v in share.items() if rg != "S6" and v < design.get("min_conforming", 0.9)]
    for name, df in (("hypotheses", H), ("structural_sensitivity", pd.DataFrame(sens)), ("model_worst_case", pd.DataFrame(model)), ("observed", pd.DataFrame(observed)),
                     ("exclusions", ex), ("channels", pd.DataFrame(chan))):
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in summ.items()},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1, default=float))
    return summ


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "audit":
        ok, lines = audit(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2); print("\n".join(lines)); sys.exit(0 if ok else 1)
    elif cmd == "gate":
        gate(sys.argv[2], sys.argv[3:])
    elif cmd == "eval":
        evaluate(sys.argv[2], json.load(open(sys.argv[3])))
