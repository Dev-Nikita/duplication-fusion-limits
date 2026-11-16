# R5 protocol — independent, block-structured confirmation on the own 5G testbed
**Frozen 2026-10-18, before any R5 episode exists.** Code hashes are listed at the end; any later change
goes to `docs/amendments.md` with a reason, and the frozen version is reported as primary.

## Why R5
R4 (one continuous run, 120 episodes) confirmed C1–C3, not C4; AUSF stalls were not learnable at weak
intensity. Episodes inside one run are not independent units, so R4 intervals may be too narrow. R5 adds
(i) independent blocks, (ii) a stronger, separately declared fault configuration, (iii) a fourth fault type,
(iv) frozen strong baselines.

## Design
* **Blocks.** 3 blocks; before each: containers down → up → re-provision (12 000 subscribers) → preflight.
  Block b uses schedule seed 20261025 + 1000·b. 20 episodes per class per block (≈ 60 per class).
* **Classes.** healthy; amf_stall and ausf_stall (SIGSTOP duty cycle U(0.30, 0.70), period 1.0 s);
  reg_reject (share of unprovisioned UEs U(0.05, 0.50)); net_delay (tc netem one-way delay U(30, 200) ms on
  traffic to the AUSF address).
* **Pre-registered fallback.** If netem is unavailable in the Docker VM the block runs without net_delay
  (logged); net_delay episodes with `netem_ok ≠ true` are excluded. If net_delay is missing in all blocks,
  the analysis is run on the other three tasks and this is reported.
* **Separation from R4.** The weak R4 AUSF configuration stays a separate, reported negative result;
  R4 and R5 episodes are never pooled. R5 is a new configuration, not a re-run of R4.
* **Features.** The frozen R4 extractors, unchanged (`experiments/r5/build_features_r5.py`). Interrupted
  episodes (no meta.json) and duplicated schedule positions are excluded and listed.

## Analysis (`experiments/r5/run_r5.py`, `evaluate_r5.py`)
* Tasks: each fault vs healthy. **Leave-one-block-out:** train + calibrate on two blocks (70/30 split
  stratified, 5 seeds), test on the held-out block. Thresholds = 90th percentile of healthy training episodes.
* **Learnability gate:** stacking AUROC on calibration ≥ 0.65; a (task, held-out block) fold is kept if
  ≥ 3 of 5 seeds pass; a task is kept if ≥ 2 of 3 folds pass. Gate uses no test data.
* Observation layer and grids as R4: D ∈ {0, 1, 2, 4, 8, 16}, h ∈ {0, .25, .5, .75, 1}. Confirmatory cells
  only (8, 0.5) and (1, 0.5) (and (16,0)/(0,0) for C3); the rest is a secondary utility map.
* **Unit of resampling:** (task, held-out block) cell, averaged over seeds. **Hierarchical bootstrap**
  (20 000): resample blocks, then tasks within a block.

## Confirmatory criteria
* **C1** Brier(stacking) − Brier(A4) at (8, 0.5): 95% CI > 0 **and** positive in each of the 3 held-out blocks.
* **C2 (non-inferiority)** Brier(A4) − Brier(stacking) at (1, 0.5): one-sided 95% upper bound < 0.01.
* **C3** |Brier_A4(16, 0) − Brier_A4(0, 0)| < 0.005 in every cell (known lineage ⇒ duplication-invariant).
* **C4 (secondary)** certified-decision error < A4 point error at (8, 0.5): CI < 0.
* **C5 (baselines)** at (8, 0.5), Brier(baseline) − Brier(A4), CI > 0, reported separately for:
  PACT-style meet with the declared partition; Taylor–Bishop-type log-linear pooling with α tuned on
  calibration by Brier (α ∈ {0.05, …, 1.0}); geometric pooling (α = 1/m); naive Bayes; generative BMA.
  PACT with the true partition and the generative oracle are reported as references, not competitors.
* Also reported: NLL, ECE, AUROC, interval width of A4, runtime; absolute Brier of every method.

## Baseline adaptation (stated in the paper)
* PACT (Jin et al., arXiv 2609.01662) has no public code: re-implemented from the paper — evidence e = (max(0, llr),
  max(0, −llr)); meet (component-wise minimum) inside a declared provenance component, sum across components.
  Declared partition: hidden readings and copies are singletons. Property tests: `tests/test_pact_properties.py`.
* Taylor–Bishop-type pooling: the log-linear (generalised geometric) pool of per-origin likelihood ratios
  with one global exponent α; for binary hypotheses with conditionally dependent sources this is the
  conservative family; we tune α rather than derive it from a dependence model we do not have. This
  adaptation is stated as such, not as a reproduction of the original estimator.

## Stopping and reporting
No interim looks at R5 metrics. All criteria are reported, holding or not. A dry run on R4 data with
pseudo-blocks (`--dryrun`) was used only as a code test and is not evidence.

## Frozen code (sha256)
```
f30dfe7bd6d2e90ef503d3d060d1a24d58a4cb9c480027354e2f7f0c1d027280  experiments/r5/run_r5.py
e6b5e0987a8979eb2c54d9bd949681a2e846e7bb770f5a055257c074c7927430  experiments/r5/evaluate_r5.py
fbb9715ba41c058f9d82cf8d7da8a3b7f7ae90b8b3500d3acd09cba066a4149c  experiments/r5/build_features_r5.py
ad9a7151d7b8b3cd691446a2b21b24e4cef3c9d020ff953c314275582b214382  experiments/r4/build_features.py
2d61611e06f6d615e4d43d6b0f88ec512e8e411fa13ba550de115a4bee4a6c3d  src/paef/real/generic.py
d923970b8ca7584e5a4357135c876a0b4e9777d9c647dbbfc1853163e1d8a184  testbed/scripts/campaign.py
```
