"""Duplication mechanisms for R6 (frozen before R6 data). Each returns a Window whose copies are derived
re-emissions of firing per-UE alarms; lineage of secondary readings and of copies is hidden w.p. h.
Training-time augmentation uses only 'iid'; all others are unseen at training."""
from __future__ import annotations
import numpy as np
from paef.real.generic import Window

UNSEEN = ("burst", "source", "incident", "storm", "mixed")
SOURCE_D = (1, 16, 1, 16)          # per reading type: UE-log channels rarely re-emitted, AMF-log channels heavily


def n_copies(mech, k, y, rng, D=8, storm_on=False):
    if mech == "iid":
        return D
    if mech == "burst":           # most alarms not re-emitted, some in long bursts (mean ~ 6.4)
        return 0 if rng.random() < 0.6 else int(rng.geometric(1 / 16))
    if mech == "source":
        return SOURCE_D[k]
    if mech == "incident":        # alert storms during incidents: re-emission rate depends on the state
        return 12 if y == 1 else 2
    if mech == "storm":           # window-level on/off storm shared by all alarms of the window
        return 16 if storm_on else 0
    if mech == "mixed":
        return n_copies(str(rng.choice(["iid", "burst", "source", "incident"])), k, y, rng, D=int(rng.choice([1, 4, 8])))
    raise ValueError(mech)


def make_window_mech(z, y, mech, h, rng, D=8):
    w = Window(y=y, z=z)
    storm_on = bool(rng.random() < 0.5)
    for o, zz in z.items():
        w.hidden_read[o] = [k for k in range(1, len(zz)) if rng.random() < h]
        for k in range(len(zz)):
            if zz[k] == 1:
                for _ in range(n_copies(mech, k, y, rng, D, storm_on)):
                    w.copies.append((o, k, bool(rng.random() < h)))
    return w


def lineage_separated_features(layout, w):
    """Recorded units, recorded-copy counts and unrecorded-copy counts per reading type, kept apart so the
    learner can weight each provenance class (used by the 'la_aug' method)."""
    known = np.zeros(layout.K); hid = np.zeros(layout.K)
    for _, k, hd in w.copies:
        (hid if hd else known)[k] += 1
    return np.concatenate([layout.unit_features(w), known, hid])
