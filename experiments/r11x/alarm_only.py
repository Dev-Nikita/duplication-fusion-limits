"""R11-X (EXPLORATORY, after unblinding R11): same frozen R11 pipeline, but every feature map keeps only the
alarm column (messages reporting a firing reading). The R10/R11 count features also contain the count of
non-firing readings, which copies never touch; with a fixed number of UEs per window this count reveals the
number of firing units exactly, i.e. the "lineage-free" features carry the unit count. Usage:
python3 -I alarm_only.py <units_cond.csv> <out_dir> [n_seeds]"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r11")); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "r5b")); sys.path.insert(0, os.path.join(HERE, "..", "r10"))
import run_r11 as R  # noqa: E402
from run_r5b import L  # noqa: E402
import paef.real.mechanisms as MM  # noqa: E402
K = L.K
_cnt, _unit, _ls = L.count_features, L.unit_features, MM.lineage_separated_features
L.count_features = lambda w: _cnt(w).reshape(K, 2)[:, 1].copy()
L.unit_features = lambda w, hidden_copies_as="exclude": _unit(w, hidden_copies_as).reshape(K, 2)[:, 1].copy()
L.presence_features = lambda w: (_cnt(w).reshape(K, 2)[:, 1] > 0).astype(float)
R.unit_feats = lambda w: L.unit_features(w)
if len(sys.argv) > 3:
    R.N_SEEDS = int(sys.argv[3])
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
R.main()
