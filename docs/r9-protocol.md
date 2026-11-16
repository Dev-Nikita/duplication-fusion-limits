# R9 protocol — non-linear baseline and lineage errors (secondary)
**Frozen 2026-10-19 before any R9 model was fitted.** Data: R6 episodes (blocks {4,5,6,10,11,12}); training
windows identical to R6. Secondary, post-acquisition analysis. Unit = held-out block (n = 6), 95% t-interval.
## (a) Non-linear lineage-free baseline
gbdt_aug: gradient-boosted trees on counts, augmented training, hyper-parameters chosen on augmented calibration
windows from a fixed grid (depth {2,3} × learning rate {0.05,0.1} × iterations {100,300}); gbdt_la: the same on
lineage-separated features with the same budget.
* G1: Brier(gbdt_aug) − Brier(la_aug), unseen mechanisms, h = 0.5 (is a stronger lineage-free learner enough?).
* G2: Brier(gbdt_aug) − Brier(gbdt_la) (does lineage still help a non-linear learner?).
## (b) Lineage errors (test time only; models unchanged)
* false split (recorded copy reported as a new unit) and false merge (true unit reported as a recorded copy)
  at ε ∈ {0, 0.05, 0.1, 0.2, 0.4}; augmented stacking is unaffected (it ignores lineage).
* Reported: la_aug − augmented gain per ε and error type; break-even ε where the gain crosses zero.
* MNAR: lineage unrecorded w.p. 0.75 in incident windows and 0.25 otherwise (mean ≈ MCAR 0.5).
No criterion here decides anything about R6; all are reported regardless of direction.

## Frozen code (sha256)
```
4c11eb7a2d36fe761accc2609ef278e6e10cef824cf004f2caa1482d2ca27aa6  experiments/r9/run_r9.py
```
