# R11 results (confirmatory, protocol docs/r11-protocol.md, frozen 2026-10-22)

## Data, integrity, manipulation check
* Blocks 21–32, ep_1214–ep_2413: 1 200 episodes. Each block has 50 control and 50 loaded episodes, and
  each class has 120 episodes per condition. 23 991 UE rows.
* Non-reject episodes with > 50 % rejected UEs: 0. Duplicate IMSI within a block: 0. netem applied in every
  net_delay episode. All NFs alive after every episode. Code sha256 identical to the frozen list (Mac and
  container).
* Manipulation check:

  | Condition | Docker VM CPU busy, mean (min–max) | Healthy-episode median UE latency, median / q90 |
  |---|---|---|
  | control | 0.12 (0.08–0.26) | 0.175 / 0.239 s |
  | loaded (W = 10) | 0.99 (0.96–1.00) | 1.217 / 1.885 s |

* The load made the task much harder. Exact-lineage GBDT Brier, mean over mechanisms: 0.033 (control) vs
  0.125 (loaded). Calibration AUROC under load: amf_stall 0.97, ausf_stall 0.97, net_delay 0.74.

## Pre-declared hypotheses (unit = block, n = 12, paired)
| | Estimate | 95 % t-interval | One-sided 95 % lower bound | Blocks > 0 | Verdict |
|---|---|---|---|---|---|
| H1: I = Δ_loaded − Δ_control (mix oracle) | +0.0004 | [−0.0010, 0.0019] | −0.0008 | 8/12 | **not confirmed** |
| H2: I_LD − I_LI (mix oracle) | +0.0010 | [−0.0004, 0.0024] | −0.0001 | 8/12 | **not confirmed** |
| Practical relevance (I ≥ 0.005) | — | — | — | — | **no** |

Δ_control = 0.0004 and Δ_loaded = 0.0008: the worst-case LD loss of the strongest lineage-free learner stays
below 0.001 Brier in both conditions. Secondary (no test): with gbdt_aug, I = +0.0008 [−0.0012, 0.0028] and
H2 = +0.0011 [−0.0006, 0.0028]. The type-level deduplication baseline is much worse than every other method
(Brier 0.22 control, 0.14 loaded). It discards how many units fire, so its contrasts are not informative.

## Decision (frozen rule)
H1 not confirmed → boundary paper with R9, R10 and R11 reported as they are. **No R12.**

## What R11 settles
* The R6-based pilot pattern (a positive LD gap where healthy latency is noisy) does **not** replicate under
  experimental manipulation. Under load the task is about four times harder in Brier, with real headroom, and
  the lineage benefit is still < 0.001.
* So the R10 null is **not** a ceiling effect. That explanation is now refuted directly, not just unproven.

## Post-hoc finding (exploratory; see docs/r11x-protocol.md for the confirmatory replication)
The feature maps of R5b–R11 (and of R8) give the lineage-free learners, per reading type, the count of
firing messages **and** the count of non-firing readings. Copies only add firing messages. With a known number
of reporting UEs (19–20 per window), the non-firing count reveals the number of firing units exactly. The
"lineage-free" features therefore contained the unit count through the denominator. Same R11 data and
pipeline, firing column only (experiments/r11x/alarm_only.py, 2 seeds):

| Channel | Method | LD worst-case loss vs exact lineage, control | loaded |
|---|---|---|---|
| firing + non-firing (as frozen) | gbdt_aug | +0.002 (0/12 blocks > 0.005) | +0.003 (2/12) |
| firing only | gbdt_aug | **+0.159** (12/12) | **+0.038** (10/12) |
| firing only | logistic on counts | **+0.276** (12/12) | **+0.142** (12/12) |
| firing only | gbdt_la (lineage-separated) | +0.002 (1/12) | +0.003 (2/12) |

Without the denominator, the LD-trained mixture oracle also fails on LI mechanisms it was not trained on
(firing only, control: li_source 0.469 and li_burst 0.341, vs 0.033 for exact lineage).
**Consequence for R8–R11:** the nulls show that lineage is redundant *when non-firing reports from known units
are observed*. They do not show that lineage is useless. A firing-alert pipeline (Alertmanager-style)
delivers firing alerts only.
