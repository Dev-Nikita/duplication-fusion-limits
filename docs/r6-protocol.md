# R6 protocol — does recorded provenance pay off under duplication-MECHANISM shift?
**Frozen 2026-10-18, before any R6 episode exists.** Hashes at the end; changes → amendments.md.

## Question
R5b showed that a lineage-free stacker trained across duplication *levels* (augmentation) beats A4 when
the deployment duplication mechanism equals the simulated one. R6 asks whether recorded lineage adds
value when the deployment mechanism differs from the one used for augmentation. The test is not built
around one favourable scenario: five unseen mechanisms are fixed now and the primary metric averages
over all of them, each reported separately.

## Data
New, independent campaign: six blocks ({4, 5, 6, 10, 11, 12} after amendment A7; originally 4–9), each preceded by container restart and re-provisioning,
10 episodes per class per block (healthy, amf_stall, ausf_stall, reg_reject, net_delay), same fault
configuration as R5 (R6 is therefore also an independent replication). Schedule seed 20261025 + 1000·b.
R5 data are not used in R6. Unit layer: experiments/r5b/build_units.py, unchanged.

## Duplication mechanisms (src/paef/real/mechanisms.py)
* Training augmentation (all learned baselines that augment): iid, D ∈ {0,1,2,4,8}, h ∈ {0,…,1}.
* Test, control: iid D = 8.
* Test, unseen: **burst** (60% of alarms not re-emitted, others geometric bursts, mean 16);
  **source** (UE-log readings ×1, AMF-log readings ×16); **incident** (×12 during incidents, ×2 otherwise —
  repetition carries information); **storm** (window-level on/off, ×16 or ×0); **mixed** (per alarm a random
  draw of iid/burst/source/incident with D ∈ {1,4,8}).
* Ordering/delay of copies does not affect window-level counts and is therefore not modelled; stated as a
  limitation.

## Provenance availability regimes
Recorded lineage at training and deployment (A4, la_aug, gen_bma, PACT); none (stacking, augmented,
presence, naive Bayes, pooling); matched = trained on the test mechanism (oracle, reference only). The
fraction of unrecorded lineage h ∈ {0, .25, .5, .75, 1} gives the availability curve.

## Methods
stacking (D=1), augmented (iid augmentation), A4 (midpoint), A4-incl (hidden copies counted fully, no
half-count), **la_aug** (lineage-separated features: recorded units, recorded-copy counts and unrecorded-copy
counts per reading type, same augmentation as augmented), generative BMA, naive Bayes, tuned log-linear
pooling, PACT-style meet (declared), presence, matched.

## Evaluation
Leave-one-block-out (train/calibrate on five blocks, test on the sixth), 3 seeds of the 70/30 split.
Gate: stacking calibration AUROC ≥ 0.65; fold kept if ≥ 2/3 seeds pass; task kept if ≥ 4/6 folds.
**Unit of inference: the held-out block** (tasks and seeds averaged within the block) → 6 values;
t-interval over blocks and sign count; block bootstrap secondary.

## Criteria
* **P1 (primary)** Brier(augmented) − Brier(la_aug), averaged over the five unseen mechanisms at h = 0.5:
  95% t-interval > 0 and positive in ≥ 5 of 6 blocks.
* **P2** non-inferiority on the control (iid, h = 0.5): one-sided 95% upper bound of
  Brier(la_aug) − Brier(augmented) < 0.01.
* Secondary: A4 vs augmented; generative BMA vs augmented; la_aug vs generative BMA; la_aug vs matched;
  A4-incl vs A4; per-mechanism results; availability curve over h; decided-share error; absolute Brier.

## Pre-declared consequences
* P1 and P2 hold → the paper is centred on lineage-separated learned fusion as robust to duplication-
  mechanism shift, with the boundary (where augmentation suffices) reported.
* P1 fails → the paper reports the boundary result: augmentation suffices under the tested mechanisms;
  provenance is not claimed to pay off. No further campaign is run to rescue the claim.

## Frozen code (sha256) — updated by amendment A7 (BLOCKS constant, run.sh provisioning guard)
```
5c6529000436c3306f652a80cf1cbbe941ce25bffb87e07026688895c6678267  experiments/r6/run_r6.py
4b68bb4129d6a734c9e1732584b95d70280bcf835a4d4c041f830da9a38ff5a5  experiments/r6/evaluate_r6.py
f36b09be737d9cf1408674acb4ea7b234a2cdfb2ce88282f26bf6700d1b57aec  src/paef/real/mechanisms.py
dc6a7aba1311ba4410e2e4b65d357b1e5d96dfd5286bc5226eb93dd9ee0f9531  experiments/r5b/build_units.py
d6b51f8b9abc10749d0c8b46320cb7b8cea781fb310b7c68da8e6bbeae1a9c38  experiments/r5b/run_r5b.py
8ae05c4bdcba2b8eab42cc5ec722d1ec42987e6047340ca8ea72cb9a2792a0eb  src/paef/real/units.py
2d61611e06f6d615e4d43d6b0f88ec512e8e411fa13ba550de115a4bee4a6c3d  src/paef/real/generic.py
d923970b8ca7584e5a4357135c876a0b4e9777d9c647dbbfc1853163e1d8a184  testbed/scripts/campaign.py
1b2e8cd050a9ff055cfa0e17eadd9b9e90f4a1c0727f752cc95a58ca20371289  testbed/run.sh
```
