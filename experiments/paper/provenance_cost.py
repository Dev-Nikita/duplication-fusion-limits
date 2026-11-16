"""Cost side of 'does provenance pay off': messages per window, share of derived messages, lineage metadata
bytes (parent id 16 B + lineage type 1 B per derived message, recorded fraction 1-h), and per-window
fusion time for augmented vs la_aug. R6 data, healthy+fault windows, h = 0.5. Descriptive only."""
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../../src")); sys.path.insert(0, os.path.join(HERE, "../r5b"))
from paef.real.mechanisms import make_window_mech, lineage_separated_features, UNSEEN
from run_r5b import L, episodes, fit
u = pd.read_csv(os.path.join(HERE, "../../results/r6/units.csv")).replace([np.inf], 1e9)
u = u[u.block.isin([4, 5, 6, 10, 11, 12])].copy(); u["y"] = (u.label != "healthy").astype(int)
h0 = u[u.y == 0]; thr = (float(np.percentile(h0.ue_lat, 90)), float(np.percentile(h0.amf_lat, 90)))
E = episodes(u, thr); rng = np.random.default_rng(0)
rows = []
for mech in ("iid",) + UNSEEN:
    W = [make_window_mech(z, y, mech, 0.5, rng) for _, y, z in E]
    n_msg = np.array([sum(len(zz) for zz in w.z.values()) + len(w.copies) for w in W])
    n_cp = np.array([len(w.copies) for w in W]); n_rec = np.array([sum(1 for c in w.copies if not c[2]) for w in W])
    y = [w.y for w in W]
    au = fit([L.count_features(w) for w in W], y); la = fit([lineage_separated_features(L, w) for w in W], y)
    t0 = time.perf_counter(); [au.predict_proba(L.count_features(w)[None]) for w in W]; ta = (time.perf_counter() - t0) / len(W)
    t0 = time.perf_counter(); [la.predict_proba(lineage_separated_features(L, w)[None]) for w in W]; tl = (time.perf_counter() - t0) / len(W)
    rows.append(dict(mech=mech, msgs_per_window=n_msg.mean(), derived_share=(n_cp / n_msg).mean(),
                     lineage_bytes_per_window=17 * n_rec.mean(), us_augmented=1e6 * ta, us_la_aug=1e6 * tl))
df = pd.DataFrame(rows).round(3); print(df.to_string(index=False))
df.to_csv(os.path.join(HERE, "../../results/r6/provenance_cost.csv"), index=False)
