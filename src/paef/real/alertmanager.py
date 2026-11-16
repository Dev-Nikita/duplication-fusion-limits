"""Structural emulation of documented Prometheus/Alertmanager behaviour on per-UE alarms (R8).
Frozen before any R8 model was fitted. Sources: prometheus.io/docs/alerting/latest/configuration/ and
/alertmanager/ (defaults group_wait 30s, group_interval 5m, repeat_interval 4h; 'continue' routes; inhibit_rules
with 'equal'; webhook payload carries a per-alert 'fingerprint'; max_alerts truncation; Prometheus must send to
every Alertmanager instance).
Alerts: one per (UE, reading type) that fires; labels {alertname=type, imsi}. With the documented defaults no
timing-based re-notification can occur inside a 20 s window, so only structural behaviour is emulated.
Lineage is 'recorded' for the diagnoser only when a message carries the fingerprint of an alert it has
already seen (webhook receivers); text receivers (chat, e-mail) carry no fingerprint."""
from __future__ import annotations
import numpy as np
from paef.real.generic import Window

LAT, REJ = (0, 1), (2, 3)      # reading types: ue_slow, amf_slow | ue_rej, amf_rej
SCENARIOS = ("am_default", "am_ha_replica", "am_multiroute", "am_inhibit", "am_truncate", "am_combined")
DUP_SCENARIOS = ("am_ha_replica", "am_multiroute", "am_combined")


def inhibit(z):
    """inhibit_rules: source = any rejection alert, target = latency alerts, equal: [imsi]."""
    out = {}
    for o, zz in z.items():
        zz = zz.copy()
        if zz[REJ[0]] or zz[REJ[1]]:
            zz[LAT[0]] = 0; zz[LAT[1]] = 0
        out[o] = zz
    return out


def make_window_am(z, y, scenario, rng, max_alerts=5):
    """Window seen by a diagnoser that consumes the notifications of the given pipeline.
    w.copies  : deliveries the diagnoser can tell are not new alerts, but not which alert they copy
                (chat / e-mail deliveries without fingerprint) -> unrecorded lineage;
    w.invisible: duplicates that look like new alerts (HA replicas with a distinct 'replica' label, hence a
                distinct fingerprint) -> counted as units by every method; nobody can remove them."""
    if scenario in ("am_inhibit", "am_combined"):
        z = inhibit(z)
    if scenario == "am_truncate":                 # webhook max_alerts per group (group_by: [alertname])
        z = {o: zz.copy() for o, zz in z.items()}
        for k in range(4):
            firing = [o for o, zz in z.items() if zz[k] == 1]
            for o in list(rng.permutation(firing))[max_alerts:]:
                z[o][k] = 0
    w = Window(y=y, z=z); w.invisible = []
    for o in z:
        w.hidden_read[o] = []                     # alerts carry their own labels: reading lineage is known
    for o, zz in z.items():
        for k in range(len(zz)):
            if zz[k] != 1:
                continue
            replica = scenario in ("am_ha_replica", "am_combined")
            routes = scenario in ("am_multiroute", "am_combined")
            if replica:
                w.invisible.append((o, k))
            if routes:                            # continue: true -> webhook + chat + e-mail
                n = 2 * (2 if replica else 1)     # chat and e-mail copies of the alert (and of its replica)
                w.copies += [(o, k, True)] * n
    return w


def _add_invisible(f, w, K):
    f = f.copy()
    for _, k in getattr(w, "invisible", []):
        f[2 * k + 1] += 1
    return f


def count_features_am(L, w):
    return _add_invisible(L.count_features(w), w, L.K)


def unit_features_am(L, w, hidden="exclude"):
    return _add_invisible(L.unit_features(w, hidden), w, L.K)


def la_features_am(L, w):
    from paef.real.mechanisms import lineage_separated_features
    f = lineage_separated_features(L, w); f[:2 * L.K] = _add_invisible(f[:2 * L.K], w, L.K)
    return f


def a4_am(L, w, lr):
    coef = lr.coef_.ravel(); b0 = lr.intercept_[0]
    base = float(b0 + unit_features_am(L, w) @ coef); lo = hi = base
    for _, k, hid in w.copies:
        if hid:
            lo += min(0.0, coef[2 * k + 1]); hi += max(0.0, coef[2 * k + 1])
    sig = lambda x: float(1 / (1 + np.exp(-x)))
    return sig(float(b0 + unit_features_am(L, w, "half") @ coef)), sig(lo), sig(hi)
