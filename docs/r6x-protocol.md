# R6-X protocol — does the la_aug gain come from provenance or from capacity? (secondary analysis)
**Frozen 2026-10-19 before any ablation model was fitted.** Data: the R6 units of blocks {4,5,6,10,11,12}
(already analysed in R6, so this is a secondary, post-acquisition analysis, not a confirmation).
Code: experiments/r6x/run_ablation.py, evaluate_ablation.py (hashes below). It reproduces the R6 training
and test windows exactly (same random streams); a separate stream drives the placebo and the tuning windows.

## Ablations (same training windows, same logistic model, same regularisation unless stated)
* **la_placebo** — identical feature layout to la_aug (units per type×value; recorded-copy and
  unrecorded-copy counts per type), but the recorded/unrecorded copy labels are given to randomly chosen
  messages of the window, preserving their numbers. Real parent information is destroyed; label frequencies
  and dimensionality are kept.
* **aug_cap** — augmented stacking with as many features as la_aug (counts, log1p and sqrt of firing counts),
  no provenance.
* **aug_tuned / la_tuned** — both with C ∈ {0.01, 0.1, 1, 10} chosen by Brier on augmented calibration windows.

## Criteria (unseen mechanisms, h = 0.5, unit = held-out block, 95% t-interval, n = 6)
* A1 Brier(la_placebo) − Brier(la_aug) > 0, ≥ 5/6 blocks positive  → the gain needs real lineage.
* A2 Brier(aug_cap) − Brier(la_aug) > 0, ≥ 5/6 blocks             → the gain is not extra capacity.
* A3 Brier(aug_tuned) − Brier(la_tuned) > 0, ≥ 5/6 blocks          → the gain survives equal tuning.
Reported in every case: per task, excluding reg_reject (descriptive), and a check that the reproduced
augmented − la_aug equals the R6 P1 estimate.
## Pre-declared consequence
If A1 fails, the paper may not attribute the R6 gain to provenance; it is described as a representation
effect. If A2 or A3 fails, the gain is described as not robust to capacity or tuning.

## Frozen code (sha256)
```
6a80c635ac59aa07e826901335c2a99fce4401519c662e4dd4a18d6ea401a136  experiments/r6x/run_ablation.py
e7140344be54009b805791b90ff29dc6ea623ef53470bff8a474f7ae0187eb20  experiments/r6x/evaluate_ablation.py
```
