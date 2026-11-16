"""Properties PACT states for its operator, checked on the re-implementation."""
import numpy as np
from paef.real.generic import OriginModel, Window, pact_meet


def _models():
    return {"A": OriginModel(np.array([0.2, 0.8]), np.array([[0.1, 0.9], [0.2, 0.7]])),
            "B": OriginModel(np.array([0.3, 0.6]), np.array([[0.2, 0.8], [0.1, 0.6]]))}


def test_copy_within_component_does_not_amplify():
    m = _models(); w = Window(y=1, z={"A": np.array([1, 1]), "B": np.array([0, 1])})
    p0 = pact_meet(w, m, 0.5)
    for k in range(1, 6):
        w.copies.append(("A", 0, False))
        assert abs(pact_meet(w, m, 0.5) - p0) < 1e-12   # insertion non-amplification (declared copy)


def test_unrecorded_copy_counts_as_new_unit():
    m = _models(); w = Window(y=1, z={"A": np.array([1, 1]), "B": np.array([0, 1])})
    p0 = pact_meet(w, m, 0.5)
    w.copies.append(("A", 0, True))
    assert pact_meet(w, m, 0.5) > p0                       # PACT's stated failure mode
    assert abs(pact_meet(w, m, 0.5, partition="true") - p0) < 1e-12


def test_singleton_fidelity():
    m = _models(); w = Window(y=1, z={"A": np.array([1, 0])}, hidden_read={"A": [1]})
    # each singleton component keeps its own evidence
    l = [np.log(m["A"].marg(0)[1] / m["A"].marg(0)[0]), np.log((1 - m["A"].marg(1)[1]) / (1 - m["A"].marg(1)[0]))]
    expect = 1 / (1 + np.exp(-(l[0] + l[1])))
    assert abs(pact_meet(w, m, 0.5) - expect) < 1e-12
