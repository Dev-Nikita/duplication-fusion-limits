"""R11-X confirmatory replication (docs/r11x-protocol.md). Applies the frozen R11 pipeline to data never
analysed with alarm-only features, as a single condition, under two observation channels:
  full  = R10/R11 feature maps (alarm AND non-alarm reading counts per reading type)
  alarm = alarm column only (the channel a firing-alert pipeline delivers)
Usage: python3 -I replicate.py <units.csv> <out_dir> <channel: full|alarm> <comma-separated blocks>"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r11")); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "r5b")); sys.path.insert(0, os.path.join(HERE, "..", "r10"))
import run_r11 as R  # noqa: E402
from run_r5b import L  # noqa: E402
units, out, channel, blocks = sys.argv[1], sys.argv[2], sys.argv[3], tuple(int(b) for b in sys.argv[4].split(","))
if channel == "alarm":
    K = L.K; _cnt, _unit = L.count_features, L.unit_features
    L.count_features = lambda w: _cnt(w).reshape(K, 2)[:, 1].copy()
    L.unit_features = lambda w, hidden_copies_as="exclude": _unit(w, hidden_copies_as).reshape(K, 2)[:, 1].copy()
    L.presence_features = lambda w: (_cnt(w).reshape(K, 2)[:, 1] > 0).astype(float)
    R.unit_feats = lambda w: L.unit_features(w)
elif channel != "full":
    sys.exit("channel must be full or alarm")
u = pd.read_csv(units); u = u[u.block.isin(blocks)].copy(); u["load_cond"] = "control"
tmp = os.path.join(out, "_units_in.csv"); os.makedirs(out, exist_ok=True); u.to_csv(tmp, index=False)
R.BLOCKS = blocks; R.N_SEEDS = 3
sys.argv = [sys.argv[0], tmp, out]
R.main()
