"""Ground truth must not influence any test-time output."""
import copy
import re
from pathlib import Path

import numpy as np

from paef.calibration.conformal_sets import build_sets
from paef.calibration.scorer import LineageScorer
from paef.fusion.bounds import posterior_bounds
from paef.model.lineage import FRESH
from paef.simulation.agnostic import make_agnostic
from paef.simulation.lineage_features import FeatConfig, sample_featured

SRC = Path(__file__).resolve().parents[1] / "src" / "paef"


def test_static_no_truth_reads_in_fusion():
    for f in (SRC / "fusion").glob("*.py"):
        assert not re.search(r"\.(true_root|true_code|label)\b", f.read_text()), f


def test_static_truth_reads_in_calibration_only_in_calibration_scores():
    txt = (SRC / "calibration" / "conformal_sets.py").read_text()
    for fn in re.split(r"\ndef ", txt)[1:]:
        name = fn.split("(")[0]
        if "true_code" in fn:
            assert name in {"episode_scores", "node_scores", "covers"}, name
    txt = (SRC / "calibration" / "scorer.py").read_text()
    body = txt.split("def probs")[1]
    assert "true_code" not in body


def test_dynamic_scrambled_truth_changes_nothing():
    rng = np.random.default_rng(3)
    cfg = FeatConfig()
    train = [sample_featured(rng, cfg) for _ in range(300)]
    sc = LineageScorer().fit(train)
    for _ in range(25):
        ep = sample_featured(rng, cfg)
        S1 = build_sets(sc, ep, 0.6, typed=True)
        e1 = copy.deepcopy(ep)
        for j in e1.meta:
            e1.obs[j].candidates = S1[j] if j in S1 else e1.obs[j].candidates
        b1 = posterior_bounds(make_agnostic(e1, fresh_box=True))
        ep2 = copy.deepcopy(ep)
        for o in ep2.obs:
            o.true_code, o.true_root = FRESH, -1
        ep2.label = 1 - (ep2.label or 0)
        S2 = build_sets(sc, ep2, 0.6, typed=True)
        assert S1 == S2
        e2 = copy.deepcopy(ep2)
        for j in e2.meta:
            e2.obs[j].candidates = S2[j] if j in S2 else e2.obs[j].candidates
        b2 = posterior_bounds(make_agnostic(e2, fresh_box=True))
        assert abs(b1["lower"] - b2["lower"]) < 1e-12 and abs(b1["upper"] - b2["upper"]) < 1e-12
