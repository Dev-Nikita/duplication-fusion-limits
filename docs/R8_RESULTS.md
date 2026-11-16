# R8 — documented Alertmanager behaviour (secondary, 2026-10-19)
Protocol docs/r8-protocol.md (frozen before fitting). Data: R6 episodes; training as in R6.
* R8-P1 (la_aug vs augmented, mean over HA-replica / multi-route / combined): −0.0056 [−0.018, 0.007],
  3/6 blocks → **fails** (la_aug worse on average).
* R8-P2 non-inferiority on the default pipeline: −0.0001, upper 0.0002 → holds.
* Per configuration (la_aug gain): default 0.0001; HA replica −0.0101 (2/6); multi-route 0.0001; inhibition
  0.0001; truncation −0.0004; combined −0.0066. Declared expectation for HA replica (no help possible) was
  exceeded in the negative direction: invisible duplicates enter unit counts, which lineage-aware models trust.
* A4 vs augmented: −0.0001 … −0.0157 (worst in combined).
* Truncation (max_alerts = 5) doubles the Brier of all count-based fusers (0.21) vs matched 0.113, presence 0.128.
* Exploratory, R6 decomposition correlation: Spearman 0.476, block-bootstrap 95% [0.13, 0.59] (descriptive).
