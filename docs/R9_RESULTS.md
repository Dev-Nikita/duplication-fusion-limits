# R9 — non-linear baseline and lineage errors (secondary, 2026-10-19)
Protocol docs/r9-protocol.md (frozen before fitting). R6 episodes, R6 training windows, unseen mechanisms, h=.5.
## (a) Gradient-boosted trees, augmented, tuned on calibration windows (same budget for both feature sets)
| | Brier |
|---|---|
| augmented (logistic) | 0.1081 |
| la_aug (logistic) | 0.1051 |
| gbdt_aug (no lineage) | 0.1052 |
| gbdt_la (lineage) | 0.1057 |
G1 gbdt_aug − la_aug = 0.0000 [−0.013, 0.013] (3/6). G2 gbdt_aug − gbdt_la = −0.0005 [−0.0013, 0.0002] (1/6).
Per task: reg_reject — gbdt_aug 0.0000 (perfect) vs la_aug 0.0106; stalls/delay — gbdt slightly worse than
logistic. **The R6 gain of la_aug (concentrated in reg_reject) is reproduced by a lineage-free non-linear
learner; lineage adds nothing to that learner.**
## (b) Lineage errors (gain of la_aug over augmented; ε = error rate)
false split: 0 → 0.0029 (6/6), 0.05 → 0.0032, 0.1 → 0.0025, 0.2 → 0.0012, 0.4 → −0.0035 (break-even ≈ 0.2–0.4).
false merge: 0 → 0.0029, 0.1 → 0.0029, 0.2 → 0.0029, 0.4 → 0.0001 (more benign).
MNAR (unrecorded 0.75 in incidents / 0.25 otherwise): 0.0031 [0.0004, 0.0057], 6/6 — no harm vs MCAR.
