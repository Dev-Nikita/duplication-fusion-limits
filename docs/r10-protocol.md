# R10 protocol — where model capacity stops replacing knowledge of evidence dependence
**Frozen 2026-10-22, before any R10 episode exists.** Confirmatory. New campaign: blocks 13–18 (container restart
and re-provisioning before each), 10 episodes per class per block, same fault configuration as R5/R6, schedule
seed 20261025 + 1000·b. R4–R9 motivated the hypotheses and are not used here.

## Theory being tested (docs/theory/IDENTIFIABILITY.md; instances of known results, cited as such)
* Label-independent duplication (LI): the posterior is recoverable without lineage (Theorem B), so a flexible
  lineage-free learner should match a model on exact unit lineage.
* Label-dependent duplication (LD): storms whose probability depends on the window state in opposite directions
  (storm_inc: incidents 0.8 / healthy 0.2; storm_ben, e.g. deployment storms in healthy windows: 0.2 / 0.8). No
  lineage-free predictor can be good under both (two-point argument, Theorem A); a unit model on exact lineage
  has a mechanism-invariant risk (invariance, Theorem B′).
* Lineage errors (Theorem C): false-split harm scales with copy volume, false-merge harm does not.

## Methods (same training windows; i.i.d. augmentation; GBDT grid depth {2,3} × lr {0.05,0.1} × iters {100,300}
chosen on calibration windows, same budget for every GBDT)
augmented (logistic, counts), la_aug (logistic, lineage-separated), gbdt_aug (counts), gbdt_la (lineage-separated),
gbdt_unit (exact lineage, copies removed), gbdt_mix_oracle (counts, trained with labels on BOTH LD mechanisms —
the strongest lineage-free predictor for the pair; an upper reference for any lineage-free learner).

## Primary tasks and unit of inference
Primary: amf_stall, ausf_stall, net_delay (reg_reject is trivially separable in this testbed and is reported
separately). Gate: count-stacking calibration AUROC ≥ 0.65 in ≥ 2/3 seeds per (task, held-out block).
Unit = held-out block (n = 6), 95% t-interval and sign count.

## Pre-declared predictions
* **P1 (LD)**: worst case over the LD pair, Brier(gbdt_mix_oracle) − Brier(gbdt_unit) > 0: interval > 0 and ≥ 5/6
  blocks positive. (Also reported: gbdt_aug instead of the oracle.)
* **P2 (LI)**: Brier(gbdt_aug) − Brier(gbdt_unit), mean over LI mechanisms: one-sided upper 95% bound < 0.005.
* **P3 (ERR, ε = 0.2, label-independent storms of size 4 and 16)**: growth of false-split harm from size 4 to 16
  minus growth of false-merge harm > 0: interval > 0 and ≥ 5/6 blocks.
## Decision rule
P1 and P2 hold → the paper is rewritten around the measured boundary "capacity replaces lineage under LI
duplication but not under LD duplication". Otherwise → the empirical boundary paper is submitted as is
(with R9), and R10 is reported in it whatever the outcome.

## Frozen code (sha256)
```
3f9f5fa8c9a973ef436f2e3134ff6e3fef8bee146bc221710c0281158d2f08f7  experiments/r10/run_r10.py
22d31b24e16335fb74affc5e3bc860e88c0bb1976e497379c1a7d7c9d7828d2f  experiments/r10/evaluate_r10.py
05f820fff0f580692d31ab0a66c64e7d10728b70e10b5650bbe464bf22a247f8  src/paef/real/mechanisms_r10.py
dc6a7aba1311ba4410e2e4b65d357b1e5d96dfd5286bc5226eb93dd9ee0f9531  experiments/r5b/build_units.py
d6b51f8b9abc10749d0c8b46320cb7b8cea781fb310b7c68da8e6bbeae1a9c38  experiments/r5b/run_r5b.py
f36b09be737d9cf1408674acb4ea7b234a2cdfb2ce88282f26bf6700d1b57aec  src/paef/real/mechanisms.py
8ae05c4bdcba2b8eab42cc5ec722d1ec42987e6047340ca8ea72cb9a2792a0eb  src/paef/real/units.py
d923970b8ca7584e5a4357135c876a0b4e9777d9c647dbbfc1853163e1d8a184  testbed/scripts/campaign.py
747b5578253390ebb3787ec6125ed0b4296601a34abaee9d761ce16675d89b4c  testbed/run.sh
```
