"""R10 duplication mechanisms (frozen before R10 data).
Label-independent (LI): copy process does not depend on the incident given the alarms.
Label-dependent (LD) pair: window-level alert storms whose probability depends on the window state, in
opposite directions: storms that accompany incidents (storm_inc) and storms that accompany benign
operations such as deployments in otherwise healthy windows (storm_ben). Same storm size in both."""
from __future__ import annotations
import numpy as np
from paef.real.generic import Window

LI = ("li_burst", "li_source", "li_storm")
LD_PAIR = ("storm_inc", "storm_ben")
SOURCE_D = (1, 16, 1, 16)


def _storm_prob(mech, y):
    if mech == "li_storm":
        return 0.5
    if mech == "storm_inc":
        return 0.8 if y == 1 else 0.2
    if mech == "storm_ben":
        return 0.2 if y == 1 else 0.8
    return None


def make_window_r10(z, y, mech, h, rng, storm_size=16):
    w = Window(y=y, z=z)
    p = _storm_prob(mech, y)
    storm_on = bool(rng.random() < p) if p is not None else False
    for o, zz in z.items():
        w.hidden_read[o] = [k for k in range(1, len(zz)) if rng.random() < h]
        for k in range(len(zz)):
            if zz[k] != 1:
                continue
            if mech == "li_burst":
                n = 0 if rng.random() < 0.6 else int(rng.geometric(1 / 16))
            elif mech == "li_source":
                n = SOURCE_D[k]
            else:
                n = storm_size if storm_on else 0
            w.copies += [(o, k, bool(rng.random() < h)) for _ in range(n)]
    return w
