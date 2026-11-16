# R8 protocol — documented Alertmanager behaviour as the deployment pipeline (secondary)
**Frozen 2026-10-19 before any R8 model was fitted.** Data: R6 per-UE units, blocks {4,5,6,10,11,12}
(already used in R6 → secondary, post-acquisition analysis; not a confirmation). R6 is not changed.
Emulation: src/paef/real/alertmanager.py, rules taken from the Prometheus/Alertmanager documentation
(configuration and overview pages, read 2026-10-19). It is an emulation of documented behaviour, not a run of
a production Alertmanager.

## What is emulated, and why only structure
Alerts = per-UE alarms (labels alertname = reading type, imsi). With the documented defaults
(group_wait 30 s, group_interval 5 m, repeat_interval 4 h) no timing-based re-notification occurs inside a
20 s window, so timing is not emulated (stated as a limitation). Lineage visible to the diagnoser = the
webhook 'fingerprint'; chat/e-mail deliveries carry none. Pipeline configurations (none uses the incident
label):
* am_default — single Alertmanager, webhook receiver: deduplicated, no copies.
* am_ha_replica — two Prometheus replicas with a distinct external 'replica' label → each alert arrives twice
  with distinct fingerprints: duplicates look like new alerts (invisible to every method).
* am_multiroute — route with continue: true to webhook + chat + e-mail: two copies per alert without fingerprint.
* am_inhibit — inhibit_rules: rejection alerts suppress latency alerts of the same imsi (equal: [imsi]).
* am_truncate — webhook max_alerts = 5 per alert group.
* am_combined — replica + multiroute + inhibition.
Training: unchanged from R6 (i.i.d. augmentation); matched = trained on the scenario (oracle reference).

## Criteria (unit = held-out block, n = 6, 95% t-interval)
* R8-P1: Brier(augmented) − Brier(la_aug), mean over the duplicating configurations {am_ha_replica,
  am_multiroute, am_combined}: interval > 0 and ≥ 5/6 blocks positive.
* R8-P2: non-inferiority of la_aug on am_default (one-sided upper bound < 0.01).
* Declared expectation: on am_ha_replica la_aug cannot help (its duplicates are invisible), so a gain there
  would indicate an artefact.
* Secondary: per configuration, A4, gap to matched, per task, excluding reg_reject, absolute Brier.

## Frozen code (sha256)
```
62ac93139beed07c9a745072e238fc1254304e7c016bae8a3cca9c79eeaaabb1  experiments/r8/run_r8.py
552db0ddc4ab1a0ac2c1062da864ed35ddc821dde13320012c4161ab8a05f1eb  experiments/r8/evaluate_r8.py
68a09e4d23369755edf52d115f2106c7fc37dc258217039a089d2b6d9a0867de  src/paef/real/alertmanager.py
```
