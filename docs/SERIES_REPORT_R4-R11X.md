# Consolidated report R4–R11-X: when does knowledge of evidence lineage matter for fusion?
Status 2026-10-24. Every protocol was frozen (sha256) before its data or analysis existed, unless stated
otherwise. Negative results are reported as they are. No new testbed campaign follows this report.

## 1. Setting
* **Testbed:** Open5GS v2.7.6 + UERANSIM v3.2.7 in Docker on one Mac.
* **Faults:** AMF stall, AUSF/UDM stall, registration reject, network delay to AUSF, plus healthy.
* **Units:** UEs, about 20 per window. Each UE gives four binary readings: UE slow, AMF slow, UE rejected,
  AMF rejected.
* **Duplication:** imposed on the real readings (semi-synthetic). It is label-independent (LI: burst, source,
  storm) or label-dependent (LD: storms more likely in incidents, or in healthy windows).
* **Lineage-free learners:** logistic regression and GBDT on message counts, with i.i.d. multiplicity
  augmentation.
* **Lineage-aware learners:** GBDT on distinct units (gbdt_unit), and lineage-separated features (gbdt_la).
* **Inference:** leave-one-block-out, with the block as the unit of inference.

## 2. History of hypotheses and outcomes
| Run | Pre-declared question | Outcome | What changed afterwards |
|---|---|---|---|
| R4 | Does a lineage-aware fuser (A4) beat stacking under heavy duplication? | C1–C3 hold; C4 not confirmed; ausf_stall failed the gate | Stronger fault configuration |
| R5/R5b | Same, 3 independent blocks, per-UE units | C1–C4 hold; generative BMA better than A4 (C5 fails) | Mechanism shift studied |
| R6 | Do lineage-separated features (la_aug) help under unseen mechanisms? | P1 holds, +0.0037 Brier (6/6) | Blocks 7–9 invalid (IMSI bug, A7) |
| R6-X | Is the gain due to lineage? | Placebo test fails; described as a representation effect | — |
| R7 | Are natural repeats present in the testbed? | No (0 retries); R7 not run | — |
| R8 | Alertmanager-documented behaviour | P1 fails; HA replicas harm lineage-aware models | — |
| R9 | Does a non-linear lineage-free learner match la_aug? | Yes: gbdt_aug ≈ la_aug ≈ gbdt_la | "Capacity replaces lineage" conjecture |
| R10 | LD: lineage needed (P1); LI: not needed (P2); split/merge asymmetry (P3) | P2 holds; P1 and P3 not confirmed | Ceiling hypothesis (later refuted) |
| R11 | Does host contention increase the lineage benefit (H1)? Is it LD-specific (H2)? | Not confirmed (I = +0.0004 [−0.0010, 0.0019]) | Ceiling explanation refuted; feature flaw found |
| R11-X | Without non-firing reports, does lineage matter (X1)? Do lineage-separated features recover it (X2)? | X1 holds on R10 and R6; X2 not confirmed on either | This report |

## 3. The flaw found after R11 (disclosed in the paper)
From R5b to R11 (and in R8), the "lineage-free" features held two counts per reading type: firing messages
(A) and non-firing reports (N). Copies are added only to firing readings. Each unit gives exactly one
reading per type, no non-firing report is lost, and M = 19–20 is known. Under these conditions the number of
distinct firing units is U = M − N exactly. So the lineage-free learners could read the unit count from the
denominator. R8–R11 compared lineage against a channel that already encoded it. Their nulls show that lineage
is redundant when non-firing reports from known units are observed. They do not show that lineage is useless.
U = M − N breaks if non-firing reports are lost, if the population is unknown or varies, if non-firing
reports are duplicated, or if a unit gives several readings per window.

## 4. R11-X results
**Retrospective, protocol-frozen cross-campaign validation.** R10 and R6 had never been analysed with the
firing-only channel, but both shaped the programme. They are not independent new samples.

### 4.1 Frozen X1–X3 (worst case over the LD pair, Brier loss relative to gbdt_unit, n = 6 blocks)
| | R10 | R6 | Criterion | Verdict |
|---|---|---|---|---|
| X1 gbdt_aug, firing only | +0.122 [lb 0.060], 6/6 > δ | +0.048 [lb 0.025], 6/6 > δ | lb > δ = 0.005, ≥ 5/6 | **holds in both** |
| X2 gbdt_la, firing only | +0.0029 [ub 0.0058] | +0.0049 [ub 0.0078] | ub < δ | **not confirmed in either** |
| X3 gbdt_aug, full channel (not blind) | +0.0009 [ub 0.0039] | +0.0041 [ub 0.0075] | ub < δ | holds R10, not R6 |
| (reported) logistic, firing only | +0.268 | +0.127 | — | — |
| (reported) mixture oracle, firing only | +0.016 | +0.008 | — | — |

**Frozen decision:** X1 holds but X2 does not → no rebuild under the R11-X rule. The boundary paper is
submitted with R9–R11 and R11-X reported as they are.

### 4.2 Exploratory X4–X7 (not confirmatory; LD worst case; Brier loss relative to gbdt_unit; 95 % t-interval)
| Channel given to the lineage-free GBDT | R10 | R6 |
|---|---|---|
| A: firing only | +0.108 [0.048, 0.168], 6/6 | +0.039 [0.016, 0.061], 6/6 |
| X4 A + distinct firing units (oracle) | +0.0007 [0.0001, 0.0012], 0/6 | +0.0078 [0.0064, 0.0092], 6/6 |
| X5 A + population size M, no non-firing reports | identical to A | identical to A |
| F: A + non-firing reports | +0.0012 [−0.009, 0.011], 1/6 | +0.0071 [0.001, 0.013], 5/6 |
| X6 F with 30 % of non-firing reports lost | +0.014 [0.003, 0.025], 5/6 | +0.009 [0.003, 0.016], 4/6 |
| X7 F, unknown population M ~ U{8..20} | −0.0096 [−0.036, 0.016], 1/6 | +0.0058 [−0.005, 0.017], 3/6 |
| LA: lineage-separated (firing units, recorded and unrecorded copies) | +0.0010 [−0.001, 0.003], 0/6 | +0.0075 [0.002, 0.013], 4/6 |

Log loss shows the same ordering. Logistic regression shows the same pattern with larger losses on firing
only (+0.257 and +0.114).

**Why X1 (frozen) and the exploratory firing-only row differ (R10 0.122 vs 0.108; R6 0.048 vs 0.039).** It is the
same estimand, computed in two independent runs. Each run has its own random streams for the
calibration split, augmentation draws and duplication draws, with 3 seeds each. The per-cell seed SD of the
gap is 0.037 (R10) and 0.026 (R6), so the differences of the means (0.014, 0.009) are within Monte-Carlo
variability. Per-block agreement is high on R10 (r = 0.94) and low on R6 (r = 0.14; smaller gaps, more noise).
For the paper, the frozen X1 values are primary. Every Monte-Carlo-based number will carry its seed
SD, and the final tables will be recomputed with 10 seeds as a declared robustness check.

### 4.3 Reading
1. **Most of the lineage value is multiplicity (X4).** Adding the distinct-firing-unit count removes 99 % of
   the firing-only loss on R10 (0.108 → 0.0007) and 80 % on R6 (0.039 → 0.0078). On R6 the remainder,
   +0.0078 [0.0064, 0.0092], is above δ, so the two models are **not** shown to be equivalent there. The same
   remainder appears for every model that has at least the unit information (F +0.0071, LA +0.0075), which
   points to an estimation floor rather than missing lineage. That reading is exploratory. The supported
   statement: the result concerns the observability of evidence multiplicity, not a richer provenance
   structure.
2. **Two channels deliver that multiplicity:** explicit lineage, or non-firing reports from the same units.
   Population size alone does not (X5; weak test here, because M is almost constant).
3. **The denominator is a sufficient condition for recovering distinct-source counts here, not a proof that
   provenance structure is redundant in fusion generally.** X3 holds on R10 but not on R6.
4. **The denominator degrades under report loss:** 30 % loss costs +0.009 to +0.014 (X6). In this testbed it
   survives an unknown population (X7). Faults here affect nearly all UEs, so firing fractions are near 0 or 1;
   this is testbed-specific and should not be generalised.
5. **Why X2 failed.** Models given the same or more information than gbdt_unit (A + U, F, LA) all sit at
   +0.007–0.008 on R6. That is the noise floor of comparing two separately tuned GBDTs at a worst case over
   two mechanisms, and it is above δ. This explanation is exploratory; X2 stays "not confirmed".
6. **A mechanism-trained learner nearly recovers the LD pair on firing only (+0.016 / +0.008).** On R11 it
   fails on LI mechanisms it was not trained on (+0.34 and +0.47). Removing copies by lineage is
   mechanism-invariant; learning the mechanism is not. This is the empirical counterpart of the invariance
   argument (Theorem B′ instance).

## 5. Consequences for the manuscript
* **Central, measured message:** a lineage-free fuser is adequate when the evidence channel carries non-firing
  reports from known units. It fails by 0.05–0.27 Brier when the channel carries firing alerts only, as
  Alertmanager-style pipelines do. Explicit lineage or a distinct-source count restores it.
* **Must be stated plainly:** R8–R11 were run on the denominator channel; the flaw was found after R11; R11-X
  is retrospective; X2 not confirmed; the mechanisms are semi-synthetic; there is one testbed and one host.
* **Novelty risk:** "if you know the denominator you know the count" can be called obvious. The contribution
  has to be argued as measured sufficiency conditions for fusion channels: which channel properties make
  provenance metadata redundant, and how much is lost when they fail. This needs positioning against
  dependent-observation, missing-data and evidence-fusion literature.
* **R8 on firing-only channels — done (exploratory, experiments/r11x/explore_r8.py, results/r8x/).**
  Documented Alertmanager behaviour on R6 episodes; Brier loss of the lineage-free GBDT relative to the
  distinct-unit GBDT, 6 blocks:

  | Scenario | firing only | firing + non-firing |
  |---|---|---|
  | default / inhibition | −0.004 (1/6 > δ) | −0.004 (0/6) |
  | multi-route (deterministic chat + e-mail fan-out) | −0.002 (2/6) | −0.004 (0/6) |
  | HA replica (invisible duplicates) | −0.016 (2/6) | −0.011 (2/6) |
  | combined | **+0.020** [−0.000, 0.040] (5/6) | −0.011 (2/6) |
  | truncation (max_alerts = 5) | −0.041 | −0.014 |

  Documented Alertmanager duplication is deterministic and label-independent (fixed fan-out per route or
  replica). A flexible lineage-free learner trained with multiplicity augmentation absorbs it even without
  non-firing reports (consistent with the LI result, Theorem B instance). Only the combined configuration
  approaches δ. Logistic regression on firing-only counts is fragile: +0.068 under the default pipeline,
  through augmentation shift. **So the large firing-only losses of R11-X need label-dependent duplication
  (storms whose rate depends on the incident state). In this study that duplication is synthetic, and
  documented Alertmanager behaviour does not produce it.** The paper must not claim that real Alertmanager
  pipelines need lineage.

## 6. Not done and not claimed
No production alert data. No natural duplication (R7 found none). The equality U = M − N is exact here by
construction of the unit layer. Its breakdown is shown only for random loss (X6) and population variation (X7).
