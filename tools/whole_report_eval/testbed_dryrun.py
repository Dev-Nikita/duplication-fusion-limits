"""Exploratory dry run of the Phase-C analysis on EXISTING testbed data (not the K=2 experiment, post hoc, not a
hypothesis test). Each UE is a source whose whole report has two types: UE-side slow (f1) and AMF-side slow (f2), as in
tools/multitype_data.py. Copying is model-based (no physical copies). Leave-one-block-out folds; thresholds from the
training blocks. Only valid blocks are used (results/integrity/DATA_QUALITY_REGISTRY.csv).
Usage (from the project root): python3 -W ignore tools/whole_report_eval/testbed_dryrun.py results/whole_report_eval"""
import os, sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wr_k2 as W  # noqa: E402

M = 20; TASKS = ("amf_stall", "ausf_stall", "net_delay"); DS = (1, 2, 4, 9)
SPECS = [("R10", "results/r10/units.csv", [13, 14, 15, 16, 17, 18], None),
         ("R6", "results/r6/units.csv", [4, 5, 6, 10, 11, 12], None),
         ("R11-control", "results/r11/units_cond.csv", list(range(21, 33)), "control"),
         ("R11-loaded", "results/r11/units_cond.csv", list(range(21, 33)), "loaded")]


def episodes(sub, t1, t2):
    s = sub.copy()
    s["f1"] = (s.ue_lat > t1).astype(int); s["f2"] = ((s.amf_seen == 1) & (s.amf_lat > t2)).astype(int)
    s["p01"] = ((s.f1 == 0) & (s.f2 == 1)).astype(int); s["p10"] = ((s.f1 == 1) & (s.f2 == 0)).astype(int)
    s["p11"] = ((s.f1 == 1) & (s.f2 == 1)).astype(int)
    w = s.groupby(["episode", "y"])[["p01", "p10", "p11"]].sum().reset_index()
    return [(int(y), (int(a), int(b), int(c))) for y, a, b, c in zip(w.y, w.p01, w.p10, w.p11)]


def main(out):
    os.makedirs(out, exist_ok=True); rows = []; chan = []
    for name, path, blocks, cond in SPECS:
        u = pd.read_csv(path).replace([np.inf], 1e9); u = u[u.block.isin(blocks)]
        if cond:
            u = u[u.load_cond == cond]
        for task in TASKS:
            sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
            for b in blocks:
                tr, te = sub[sub.block != b], sub[sub.block == b]
                if te.y.nunique() < 2:
                    continue
                t1 = float(np.percentile(tr[tr.y == 0].ue_lat, 90))
                seen = tr[(tr.amf_seen == 1) & (tr.y == 0)]
                t2 = float(np.percentile(seen.amf_lat, 90)) if len(seen) else np.inf  # no healthy AMF readings: type 2 never fires
                etr, ete = episodes(tr, t1, t2), episodes(te, t1, t2)
                cnt = lambda E: {k: sum(1 for e in E if e == k) for k in set(E)}
                ptr, pte = W.law_from_counts(cnt(etr)), W.law_from_counts(cnt(ete))
                obs = [(y, n[1] + n[2], n[0] + n[2]) for y, n in etr]
                pres = W.PresenceRule(obs)
                rules = {"presence": pres, "count_table": W.CountTable(obs, pres),
                         "logistic_log": W.Logistic(obs, "log", M), "logistic_clip": W.Logistic(obs, "clip", M)}
                r = dict(dataset=name, task=task, held_block=b, n_train=len(etr), n_test=len(ete),
                         states_train=len({n for _, n in etr}), states_test=len({n for _, n in ete}))
                Vinf_te = W.minimax(pte, "whole", W.INF)[0]
                r["RB_presence_test"] = float(W.bayes(W.channel_law(pte, W.presence).values()))
                r["V_whole_inf_test"] = float(Vinf_te)
                r["H1_stat"] = r["RB_presence_test"] - r["V_whole_inf_test"]
                for D in DS:
                    fw = W.MinimaxRule(ptr, "whole", D, pres); fc = W.MinimaxRule(ptr, "coord", D, pres)
                    Vw, Vc = W.minimax(pte, "whole", D)[0], W.minimax(pte, "coord", D)[0]
                    r[f"V_whole_{D}"], r[f"V_coord_{D}"] = float(Vw), float(Vc)
                    r[f"J_oracle_{D}"] = float(Vc - Vw)
                    r[f"J_oracle_train_{D}"] = float(W.minimax(ptr, "coord", D)[0] - W.minimax(ptr, "whole", D)[0])
                    allr = dict(rules, minimax_whole=fw, minimax_coord=fc)
                    for k, f in allr.items():
                        r[f"WC_label_{k}_{D}"] = W.worst_case(f, pte, "whole", D, "label", M)
                        r[f"WC_state_{k}_{D}"] = W.worst_case(f, pte, "whole", D, "state", M)
                        if D == DS[0]:
                            r[f"obs_{k}"] = W.observed_risk(f, pte)
                    ta0 = np.array([n[1] + n[2] for _, n in ete]); ta1 = np.array([n[0] + n[2] for _, n in ete])
                    r[f"fallback_share_whole_{D}"] = float(1 - fw.covered(ta0, ta1).mean())  # test episodes (no copying) outside training reach
                    r[f"J_deploy_{D}"] = r[f"WC_label_minimax_coord_{D}"] - r[f"WC_label_minimax_whole_{D}"]
                    G, B, _, _ = W.gap_bound(ptr, D); Gt, Bt, _, _ = W.gap_bound(pte, D)
                    r[f"G_train_{D}"], r[f"B_train_{D}"], r[f"G_test_{D}"], r[f"B_test_{D}"] = map(float, (G, B, Gt, Bt))
                rows.append(r)
                for ch, risk, bits, need in W.channel_table(pte, 2, M):
                    chan.append(dict(dataset=name, task=task, held_block=b, channel=ch, robust_risk_D2=risk, bits=bits, needs=need))
                print(name, task, b, round(r["H1_stat"], 4), round(r["J_oracle_2"], 4), round(r["J_deploy_2"], 4), flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/dryrun_folds.csv", index=False)
    pd.DataFrame(chan).to_csv(f"{out}/dryrun_channels.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
