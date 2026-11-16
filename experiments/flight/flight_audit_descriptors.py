"""Descriptors of the Flight data with the airline (gold-source) sites removed (addendum R2, mode A = B).
Usage: python3 -I -W ignore experiments/flight/flight_audit_descriptors.py <clean_dir> <truth_dir> <out_dir>"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flight_io as F, flight_study as ST, flight_audit as AU  # noqa: E402
from flight_study import C, S, SC, QS  # noqa: E402

data = F.load(sys.argv[1], sys.argv[2]); rows, items = AU.build(data, "B"); y = np.array([r["y"] for r in rows])
cl = ST.clusters([it for d in items for it in items[d].values()], 0.95); sizes = collections.Counter(cl.values())
D = max(sizes.values()) - 1; mc = len(sizes); Fa = ST.features(rows, cl)
p = ST.law(y, Fa.U.values, mc); pA = ST.law(y, Fa.A.values, int(Fa.A.max()))
cov = [len(r["claims"]) for r in rows]; fire = Fa[Fa.U > 0]; mult = fire.A / fire.U
desc = dict(n_items=len(rows), n_sources=len(cl), coverage_min=min(cov), coverage_max=max(cov), n_clusters=mc,
            cluster_sizes=sorted(sizes.values(), reverse=True), D_hat=D, RB_U=S.bayes_risk_U(p), RB_P=S.bayes_risk_presence(p),
            RB_A_natural=S.bayes_risk_U(pA), R_star_Dhat=C.ld_minimax(p, D)[0], D_sat=int(np.ceil(SC.rho(p) - 1e-12) - 1),
            copies_per_firing_cluster_y0=float(mult[y[fire.index] == 0].mean()), copies_per_firing_cluster_y1=float(mult[y[fire.index] == 1].mean()))
json.dump(desc, open(os.path.join(sys.argv[3], "flight_descriptors_no_gold_sites.json"), "w"), indent=1); print(json.dumps(desc, indent=1))
