# R2 protocol — provenance utility on real 5G signalling (ORIGAMI)
**Frozen 2026-10-17 before the first scored run.** Code: `src/paef/real/origami.py`,
`experiments/r2/run_r2.py`; data: `results/r2/windows_case_study.csv` (built on the Mac by
`experiments/r2/build_windows.py` from the case-study CSVs; window = 50 consecutive UEs of one run).

## Status of the data (stated in the paper exactly like this)
Real signalling (Open5GS + UERANSIM, public). Observation layer and its lineage are constructed by us
and recorded when observations are built → **semi-synthetic in provenance**.

## Task
H = 1 (1 core per NF VM) vs H = 0 (8 cores), same arrival rate; analysed per rate (10/20/50/100 UE/s).
Splits by run inside each configuration: 8 train / 6 calibration / 6 test runs, 5 split seeds.

## Observation layer
* Origins: the 9 NF message streams of a window (amf, ausf, udm, udr, smf, upf, pcf, bsf, gnB).
* Readings of an origin (same message set): p90, mean and max service-time alarms; thresholds =
  90th percentile of 8-core TRAIN windows at that rate. The p90 alarm is the declared primary.
* Copies: every firing alarm is re-emitted D times by monitoring (D = duplication level).
* Hidden lineage: each mean/max reading and each copy has its lineage hidden with probability h.
  Typed candidates: reading → {its origin, FRESH}; copy → {COPY(its parent), FRESH}.
  In this data no hidden observation is truly FRESH (all are derived) — stated as a limitation.

## Model and methods (all fitted on TRAIN runs only)
Per origin a latent-class model (binary U, three readings, H known) by EM; FRESH = the alarm type's
own marginal P(z | h). Prior 0.5.
* A (lineage ignored): naive Bayes on marginals; geometric pooling (mean log-LR; member of the
  conservative family of Taylor & Bishop 2019, adapted); logistic stacking on alarm-type counts
  (trained at D_train = 1); PACT-style meet, declared partition only.
* B (partial lineage): declared-only exact model (hidden → FRESH); paef_bma (uniform over typed
  candidates); paef envelope [L, U] (certified decision if L > 0.5 or U < 0.5); PACT-style meet with
  candidate closure.
* C (full lineage, reference): exact model with the true lineage.
Factors: D_test ∈ {0, 1, 2, 4, 8}; h ∈ {0, 0.5, 1}; rate ∈ {10, 20, 50, 100}; 5 split seeds.

## Hypotheses (fixed now)
* R2-H1 (duplication shift): from D_test = 1 to 8, Brier of naive Bayes and of logistic stacking rises
  by more than that of C and of paef_bma at h = 0.5 (paired over seeds × rates; bootstrap CI excludes 0).
* R2-H2 (utility map, descriptive): Brier(best A) − Brier(C) over (D, h); report the region where the
  gap < 0.005 ("provenance not worth collecting") and where it > 0.02.
* R2-H3 (model adequacy): ECE of C. If ECE > 0.05 the generative model is misspecified on real data and
  certificates are not trusted on it (reported, not hidden).
* R2-H4: certified-decision share and error of the envelope (truth of the posterior is unknown on real
  data; only label error is measurable).
Nothing above is edited after the run; deviations go to docs/amendments.md.
