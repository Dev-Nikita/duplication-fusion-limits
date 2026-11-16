# R10 results (confirmatory, protocol docs/r10-protocol.md, frozen 2026-10-22)

## Data and integrity
* Blocks 13–18, episodes ep_0878–ep_1177: 300 episodes, 10 per class per block, 5 classes; 5996 UE rows.
* Non-reject episodes with >50 % rejected UEs: 0. Duplicate IMSI within a block: 0. netem applied in every
  net_delay episode. UE readings per episode: 19–20.
* Code sha256 checked against the frozen list (container and Mac): all 9 files OK.
* Gate (calibration AUROC ≥ 0.65): passed in every (task, block, seed). Mean AUROC: amf_stall 1.000,
  ausf_stall 0.999, net_delay 0.965, reg_reject 1.000.

## Pre-declared predictions (primary tasks amf_stall, ausf_stall, net_delay; unit = held-out block, n = 6)
| Prediction | Estimate | 95 % t-interval | Blocks > 0 | Verdict |
|---|---|---|---|---|
| P1: LD worst case, Brier(mix_oracle) − Brier(unit) | +0.0001 | [−0.0004, 0.0006] | 4/6 | **not confirmed** |
| (reported) LD worst case, Brier(gbdt_aug) − Brier(unit) | +0.0020 | [−0.0015, 0.0055] | 4/6 | — |
| P2: LI, Brier(gbdt_aug) − Brier(unit); one-sided upper 95 % | −0.0002; **0.0010 < 0.005** | [−0.0016, 0.0013] | 3/6 | **confirmed** |
| P3: split growth − merge growth (ε = 0.2, storm 4→16) | −0.0036 | [−0.0220, 0.0149] | 2/6 | **not confirmed** |

reg_reject (reported separately): all GBDT variants reach Brier 0.0000; differences are identically zero.

## Decision (frozen rule)
P1 does not hold → **no rewrite around the LI/LD boundary.** The empirical boundary paper is submitted with
R9, and R10 is reported in it as a confirmatory test: the LI prediction (P2) is confirmed, while the LD
prediction (P1) and the lineage-error asymmetry (P3) are not detected.

## Exploratory diagnosis (not confirmatory; added after unblinding)
* **Near-separability (a description, not a demonstrated cause).** R10 windows are almost perfectly separable.
  Mean Brier by task: amf_stall 0.0028 for every GBDT variant, ausf_stall ≈ 0.02, net_delay ≈ 0.064.
  Theorem A bounds the LD gap by ¼E(p₁ − p₂)², where p₁, p₂ are the lineage-free posteriors under the two LD
  mechanisms. That bound is small only when the unit evidence pins both posteriors to the same value;
  posteriors near 0/1 alone do not imply this. That near-separability *caused* the null P1 is not shown by
  R10; the null is also compatible with uninformative lineage, low power, or properties of the constructed
  mechanisms.
* **Pilot on R6 data (blocks 4–6, 10–12; not R10; frozen R10 code in --dryrun mode, so metrics are inspected
  here for the first time).** Same pipeline and mechanisms on harder data (unit-model Brier ≈ 0.09–0.14,
  calibration AUROC ≈ 0.83–0.96):

  | Data | Worst-case LD gap, gbdt_aug − unit (blocks > 0) | Worst-case LD gap, mix_oracle − unit (blocks > 0) |
  |---|---|---|
  | R6 blocks 4–6 only | +0.0063 (3/3) | +0.0036 (3/3) |
  | R6 blocks 10–12 only | +0.0058 (2/3) | +0.0055 (3/3) |
  | R6 all six blocks | +0.0042 (5/6) | +0.0027 (4/6) |
  | R10 (confirmatory) | +0.0020 (4/6) | +0.0001 (4/6) |

  Making windows harder by sub-sampling UEs per window (2, 4, 8, 20 UEs) does **not** increase the gap
  monotonically, so "harder task ⇒ larger lineage benefit" is not supported either. The visible difference
  between R6 and R10 is noise in healthy episodes: the healthy-window median-latency 90th percentile is
  0.87–2.0 s in R6 blocks 10–12 versus 0.35–0.66 s in R10. This is not carry-over from preceding stall
  episodes; it looks like uncontrolled host-level load. With two campaigns, this is a contrast of n = 2 and
  is only a hypothesis for a new, pre-registered design.
* P2 is a one-sided non-inferiority result (loss < 0.005) for the three LI mechanisms tested. It is not a
  two-sided equivalence and does not extend to other mechanisms.
* The duplication mechanisms are synthetic and are applied to real testbed measurements (semi-synthetic
  evidence). The results do not describe the behaviour of real production alert storms.
* The six blocks share one testbed, one configuration and one host. They are replicates with a core restart
  between them, not independent operational environments.
* The only task with headroom is net_delay. There, gbdt_aug under storm_ben is 0.0678 vs 0.0639 for
  gbdt_unit (+0.004); the mix oracle closes this gap (0.0632). That direction is consistent with the theory,
  but it is a single task, post hoc, and not tested.
* The logistic model on counts (augmented) has lower Brier than every GBDT on the primary tasks
  (e.g. net_delay ≈ 0.059 vs 0.064). The R9 finding (the la_aug gain is a capacity effect) therefore does not
  imply that more capacity is better in absolute terms on this testbed.

## Files
results/r10/units.csv, r10_cells.csv, r10_gate.csv, r10_results.json, run.log, evaluate.log.
