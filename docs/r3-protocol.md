# R3 — confirmation on an independent ORIGAMI study (frozen 2026-10-17, before processing its data)
Data: ORIGAMI **AMF stability study** (8 cores, 10 000 UEs, arrival 10/20/50/100/150/200 UE/s, 10 runs
each). Never used in R2 or A4 development. Windows built with the unchanged `build_windows.py` (50 UEs).
Task (different incident type from R2): H = 1 overload (150 and 200 UE/s) vs H = 0 nominal (50 and
100 UE/s). Only service-time statistics enter the observation layer (no arrival timestamps).
Splits by run inside each rate: 4 train / 3 calibration / 3 test; 5 split seeds.
Thresholds: 90th percentile of nominal TRAIN windows (pooled 50 + 100 UE/s).
Methods: unchanged code — stacking (D_train = 1), lineage-aware stacking A4 (`ls`), generative oracle,
paef_bma, naive Bayes. Same D and h grids as R2.
Confirmatory hypotheses:
* C1 (robustness): at D_test = 8, h = 0.5: Brier(stacking) − Brier(ls) > 0, paired bootstrap 95% CI
  over seeds excludes 0.
* C2 (no in-distribution cost): at D_test = 1, h = 0.5: Brier(ls) − Brier(stacking) < 0.01 (upper CI
  bound below 0.01).
* C3 (invariance): with full lineage (h = 0), the Brier of ls changes by less than 0.005 between
  D_test = 0 and 8.
* C4 (certificate): at h = 0.5, D = 8, the error among certified decisions is lower than the overall
  error of ls.
If C1 or C2 fails, the A4 claim is reported as not confirmed.

## Outcome of R3 (recorded 2026-10-17) — confirmation NOT valid
C1–C4 formally pass (C1 0.038 [0.034, 0.043]; C2 0.0025 [0.0016, 0.0032] < 0.01; C3 0.000; C4 −0.023
[−0.035, −0.011]) but **every method is at chance** (AUROC 0.50–0.52, oracle 0.52): overload at 8 cores
is not visible in window-level service-time alarms. Differences reflect calibration under duplication,
not diagnosis. The protocol lacked a learnability gate; R3 is therefore reported as uninformative, not
as confirmation.

# R3b/R3c — confirmation with a learnability gate (frozen 2026-10-17, before running)
Same code, D/h grids, metrics and hypotheses C1–C4 as R3, on case-study runs:
* R3c (primary, fully unused data): H = 1 for 2 cores vs H = 0 for 4 cores at each rate (no 2- or
  4-core window was used in R2/A4). Splits 8 / 6 / 6 runs per configuration, 5 seeds.
* R3b (secondary, negatives partly reused): 2 cores vs 8 cores. 8-core windows were used as negatives in
  R2/A4 development → reported as secondary only.
**Learnability gate (per seed and rate, evaluated on CALIBRATION runs before any test scoring):**
the stacking AUROC on calibration windows must be ≥ 0.65; rates failing the gate are excluded and listed.
C1–C4 are evaluated only on gated cells; if fewer than 2 rates pass, the confirmation is declared
uninformative again.

## Outcome of R3b/R3c (recorded 2026-10-17) — uninformative by the pre-declared gate
R3c (2 vs 4 cores): 0 / 20 seed×rate cells pass (calibration AUROC 0.46–0.64). R3b (2 vs 8 cores):
2 / 20 pass (0.47–0.69). Per protocol, no confirmatory claim is made from either.
Conclusion: inside ORIGAMI only the 1-core vs 8-core contrast carries diagnostic signal in window-level
service-time alarms, so ORIGAMI cannot provide an independent confirmation of A4. Confirmation moves to
data with an injected, learnable incident and naturally heterogeneous sources (own Open5GS testbed, R4).
