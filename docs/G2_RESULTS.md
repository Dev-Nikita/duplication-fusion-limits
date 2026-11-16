# G2 results (2026-10-17) — calibrated lineage sets and posterior envelopes

Protocol: docs/g2-protocol.md (declared before the runs). Theory: docs/theory/G2_coverage.md.
Raw: results/g2/exp_L.json, exp_P.json, exp_S.json, amendment_a3.json (exploratory).
All fusion uses leakage-free parameters (A2c). Tests: 39 pass, incl. static + scrambled-truth leakage tests.

## Key table (Experiment P, δ = 0.05, 20 repetitions × 300 test episodes; mean [2.5%, 97.5%] over repetitions)
| Method | hide | lineage cov. | posterior cov. | width U−L | decided | time/ep |
|---|---|---|---|---|---|---|
| cma_full (vacuous prior) | 0.50 | 1.000 | 1.000 | 0.167 | 0.830 | 1.3 ms |
| marginal_sets | 0.50 | **0.908** | 0.968 | 0.119 | 0.881 | 0.4 ms |
| bonferroni_sets | 0.50 | 0.983 | 0.997 | 0.155 | 0.843 | 0.7 ms |
| paef_cal (T1) | 0.50 | 0.948 | 0.988 [0.973, 1.0] | 0.138 | 0.863 | 0.5 ms |
| paef_cal_typed | 0.50 | 0.948 | 0.989 | 0.142 | 0.855 | 0.4 ms |
| A3 posterior-targeted (exploratory) | 0.50 | — | 0.960 [0.943, 0.982] | **0.113** | 0.887 | — |
| cma_full | 0.75 | 1.000 | 1.000 | 0.238 | 0.762 | 4.5 ms |
| marginal_sets | 0.75 | **0.872** | 0.967 | 0.169 | 0.834 | 0.7 ms |
| bonferroni_sets | 0.75 | 0.974 | 0.997 | 0.220 | 0.781 | 2.1 ms |
| paef_cal (T1) | 0.75 | 0.950 | 0.993 | 0.208 | 0.794 | 1.5 ms |
| paef_cal_typed | 0.75 | 0.948 | 0.995 | 0.212 | 0.786 | 1.2 ms |
| A3 posterior-targeted (exploratory) | 0.75 | — | 0.967 [0.943, 0.992] | **0.171** | 0.832 | — |
Predictive output (Brier, hide 0.5 / 0.75): bma_scored 0.101 / 0.103; argmax plug-in 0.102 / 0.105; oracle 0.099 / 0.100.

## Against the declared hypotheses
| ID | Outcome |
|---|---|
| H1 joint lineage coverage of paef_cal / typed within [1−δ−0.01, 1−δ+1/(n+1)+0.01] | **Holds** in all 32 cells of Experiment L (200 reps each; e.g. n=250, δ=0.05: 0.950 ± 0.001). Matches T1 |
| H2 marginal sets under-cover jointly | **Holds**: 0.83 / 0.76 at δ = 0.10 (hide 0.5 / 0.75) |
| H3 Bonferroni covers but larger sets | **Holds**: coverage 0.96–0.98, mean set size 3.0 vs 2.7 |
| H4 informativeness gate (posterior cov ≥ 0.94 and ≥ 30% narrower than CMA at hide 0.5) | **Fails the target, passes the kill line**: coverage holds, but widths are only 17.3% / 12.7% (paef_cal) and 14.5% / 10.9% (paef_cal_typed) below CMA at hide 0.5 / 0.75. All percentages are ratios of mean widths, 1 − mean(method)/mean(CMA). Erratum: the first report quoted the typed figures next to the untyped widths |
| H5 typed narrower than untyped | **Fails**: typed renormalisation shrinks *lineage sets* (2.47 vs 2.68 candidates) but the *envelope* is 0.0045 wider (95% of reps). Reason: value-inconsistent copies were already excluded by the likelihood, so pruning them changes nothing in the envelope |
| H6 decisions (descriptive) | Decided share 0.83 → 0.86 (hide 0.5) at unchanged risk; still not a better abstention rule than the scored point estimate |
| S shift (descriptive) | Weaker test metadata: lineage coverage 0.948 → 0.893, posterior coverage 0.988 → 0.974. Exchangeability matters; real campaigns need recalibration per regime |

## Exploratory A3
Calibrating the envelope directly (nested conformal over the lineage threshold) gives posterior
coverage at nominal (0.960 / 0.967) and widths 32% / 28% below CMA — what H4 asked for. But this is
conformalized credal-set prediction (Javanmardi, Stutz, Hüllermeier, NeurIPS 2024) applied to a
lineage-induced envelope, and it needs the true-model posterior on calibration episodes (lineage known
by construction there). Useful method; not a new one.

## Verdict
* **G2a: done.** T1 and T2 stated and proved; the empirical joint coverage matches the theory exactly.
  Both are standard results — they make the method correct, not novel.
* **G2b: weak.** Lineage-level calibration is only 13–17% (untyped) / 11–15% (typed) narrower than the vacuous credal set.
  Posterior-targeted calibration reaches ~30% but borrows a published idea.
* The typed structure (copy / reading / fresh) did **not** buy informativeness in this pilot.
* After G2, no component on its own is a new fusion *method*. What remains: the problem formulation,
  lineage queries (G3, untested beyond the pilot), and a real 5G evaluation with lineage known by
  construction. Decide the venue after G3.
