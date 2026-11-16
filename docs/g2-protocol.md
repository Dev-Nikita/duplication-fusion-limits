# G2 pilot protocol — declared 2026-10-17, before the first run

Code: `experiments/g2/run_g2.py`; seed 20261018. Synthetic, generator `simulation/lineage_features.py`
(full typed candidate universe; per-candidate metadata features; 30% of unresolved observations are
truly independent). All fusion uses the leakage-free parameterisation (agnostic + FRESH box, A2c).

## Methods (rows of the key table)
1. oracle — true lineage and parameters (reference)
2. cma_full — every candidate in the typed universe (= credal model averaging, vacuous prior)
3. marginal_sets — per-observation split-conformal sets at level 1−δ (no joint correction)
4. bonferroni_sets — per-observation sets at level 1−δ/|J| (|J| = max unresolved per episode in calibration)
5. paef_cal — episode-level max-score sets (T1)
6. paef_cal_typed — 5 with scores renormalised over value-consistent candidates
Predictive output (separate): scorer-weighted model average (`bma_scored`), argmax plug-in.

## Experiment L (lineage coverage only; cheap)
200 repetitions × n_cal ∈ {250, 500, 1000, 2000} × δ ∈ {0.10, 0.05}; n_train = 2000, n_test = 1000.
## Experiment P (posterior envelopes)
20 repetitions; n_cal = 1000; δ = 0.05; hide ∈ {0.5, 0.75}; n_test = 300.
## Experiment S (shift, descriptive)
As P at hide 0.5, but test-time metadata is weaker (true-candidate lag mean 1.0 → 1.6).

## Hypotheses and criteria (fixed now)
* H1: paef_cal and paef_cal_typed: mean joint lineage coverage over repetitions lies in
  [1−δ − 0.01, 1−δ + 1/(n+1) + 0.01] for every n_cal (Experiment L).
* H2: marginal_sets under-covers jointly (mean < 1−δ − 0.01) at hide ≥ 0.5.
* H3: bonferroni_sets covers but has larger mean set size than paef_cal (Experiment L).
* H4 (informativeness — the gate): in Experiment P, paef_cal_typed has posterior-envelope coverage of
  p* ≥ 1−δ − 0.01 **and** mean width (U−L) at least 30% below cma_full at hide 0.5.
  **Kill: < 10% width reduction ⇒ G2 not informative.**
* H5: paef_cal_typed width < paef_cal width (typed renormalisation helps) — paired, CI excluding 0.
* H6: decisions — at equal coverage of decided episodes, report risk for envelope rule vs bma_scored
  (descriptive; no claim fixed, because K1 from the pilot already shows envelopes are not a better
  abstention rule).
* S: report how far coverage falls under shift; no claim.
Runtime is reported per episode for envelope computation.
Nothing here is edited after the run; deviations go to docs/amendments.md.
