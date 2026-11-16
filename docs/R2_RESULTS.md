# R2 / A4 / R3 — real-data results (ORIGAMI), 2026-10-17
Raw: results/r2/r2_cells.csv (R2), a4_cells.csv (A4), r3_cells.csv, r3b_*, r3c_* (confirmation attempts).
Observation layer: 9 NF origins × 3 readings (p90/mean/max alarms) per 50-UE window; copies = monitoring
re-emissions (D); hidden lineage fraction h. Lineage recorded at construction (semi-synthetic provenance).

## R2 (protocol frozen before the run), 1 vs 8 cores, 5 seeds × 4 rates, means
| | D=1 | D=8 | change |
|---|---|---|---|
| naive Bayes (Brier) | 0.225 | 0.295 | +0.071 [0.053, 0.090] |
| lineage-blind stacking (trained D=1) | 0.128 | 0.196 | +0.067 [0.054, 0.082] |
| generative model, true lineage (C) | 0.182 | 0.182 | 0 |
| paef_bma, h = 0.5 | 0.189 | 0.202 | +0.013 [0.009, 0.018] |
* R2-H1 **holds**: duplication shift hurts lineage-blind fusion far more than lineage-aware fusion
  (naive − paef_bma 0.057 [0.042, 0.074]; stacking − paef_bma 0.054 [0.043, 0.065]).
* R2-H3 **triggered**: the generative lineage model is misspecified on real data (ECE 0.106 > 0.05;
  AUROC 0.77 vs 0.87 for stacking). Certificates from the generative model are not trusted on these data.
* R2-H4: generative envelope certifies 52–100% of windows but with 23–28% label error — consistent with
  the misspecification.

## A4 (exploratory): lineage-aware stacking — train on countable units, carry unresolved copies as an interval
| h | D=1: stacking / A4 | D=8: stacking / A4 | A4 certified share / error (D=8) |
|---|---|---|---|
| 0 | 0.128 / **0.126** | 0.196 / **0.126** | 1.00 / 0.18 |
| 0.5 | **0.128** / 0.130 | 0.196 / **0.157** | 0.70 / 0.15 |
| 1.0 | **0.128** / 0.133 | 0.196 / **0.180** | 0.60 / 0.15 |
D=8, h=0.5: stacking − A4 = 0.038 [0.031, 0.046]; D=1: A4 costs 0.002–0.005. Lineage makes the
discriminative model duplication-invariant when known, and degrades gracefully as it is hidden.

## Confirmation attempts inside ORIGAMI — not informative
* R3 (stability study, overload 150/200 vs 50/100 UE/s): all methods at chance (AUROC ≈ 0.50); C1–C4
  pass formally but are vacuous. Protocol lacked a learnability gate.
* R3c (2 vs 4 cores) and R3b (2 vs 8 cores) with the gate: 0/20 and 2/20 cells learnable → no claim.

## Status of the claim
A4 is a promising, **unconfirmed** result. The candidate design rule — *count units, not messages:
learn on lineage-deduplicated units and carry unresolved lineage as an interval; lineage is worth
collecting in proportion to the expected duplication shift between training and operation* — needs an
independent dataset with a learnable incident: own Open5GS + UERANSIM testbed (R4) and, optionally,
a public microservice benchmark with naturally linked logs/traces (RCAEval) as a third domain.
