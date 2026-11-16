# R11 protocol — DRAFT (not frozen). Host contention × evidence dependence
Status: draft 2026-10-22. It is frozen (with sha256 of all code and the value of W) only after the technical
load pilot has passed and the analysis code has been dry-run on R6 data. R11 is the **last** campaign: if H1
is not confirmed, no further campaign is run and the boundary paper is submitted with R9, R10 and R11.

## History that must be disclosed
R10 (confirmatory) did not confirm P1. A post-hoc pilot of the frozen R10 code on R6 data (exploratory, already
inspected) showed a positive LD gap where healthy-episode latency was noisy, and none in R10. R11 asks whether
this is reproduced when the disturbance is manipulated experimentally. R6, R10 and the pilot are not reused.

## Manipulation
Load sidecar (testbed/scripts/loadgen.py) on the same Docker VM: W busy workers (CPU spin + 64 MB memory
traffic) during an episode (warm-up 4 s), 0 otherwise. W is fixed by the technical pilot rule in
experiments/r11/pilot_check.py (manipulation check on healthy-episode latency only; no outcome model is fitted).

## Design (paired within block; chosen for power, see below)
* Blocks 21–32 (n = 12), core restart and fresh subscriber database (25 000) before each block, as in R10.
* Per block: 10 episodes per class (healthy, amf_stall, ausf_stall, net_delay, reg_reject) per condition
  (control / loaded) = 100 episodes. The order of (class, condition) is a random permutation fixed in advance
  from the block seed, so the condition is independent of the class and balanced in time.
* The condition is recorded in meta.json only. It is not a feature, and features contain no time stamps,
  block ids or episode indices.

## Analysis (code to be frozen: experiments/r11/run_r11.py; R10 pipeline otherwise unchanged)
* Leave-one-block-out. Models are trained and calibrated **within condition** on the other 11 blocks and
  tested on the held-out block's episodes of the same condition. Train and test never share an episode or
  copies derived from one. 5 Monte-Carlo seeds per (task, block, condition).
* Methods: logistic on counts (augmented), GBDT on counts (gbdt_aug), GBDT on presence features
  (deduplication baseline, gbdt_dedup), GBDT on lineage-separated features (gbdt_la), GBDT on exact lineage
  (gbdt_unit), and a GBDT trained on both LD mechanisms (gbdt_mix_oracle). Mechanisms: none, LI set, LD pair.
* Δ_c = worst case over the LD pair of Brier(gbdt_mix_oracle) − Brier(gbdt_unit) in condition c, averaged
  over the primary tasks (amf_stall, ausf_stall, net_delay).

## Hypotheses
* **H1 (primary):** I = Δ_loaded − Δ_control > 0. Confirmed if the one-sided 95 % t lower bound over the 12
  paired block differences is > 0 **and** ≥ 9/12 blocks are positive.
* **Practical relevance:** reported as meaningful only if the estimate of I is ≥ δ = 0.005. That is the margin
  already declared practically negligible in R10 (P2). It is not taken from the pilot.
* **H2 (specificity control):** I_LD − I_LI > 0, the same criteria. This separates a dependence-specific
  effect from a general loss of accuracy under load.
* Secondary, reported without a test: the same contrasts with gbdt_aug and gbdt_dedup in place of the
  oracle; log loss, ECE, AUROC; the manipulation check (healthy latency, VM CPU busy, docker stats per block).

## Power (from data already seen; approximate)
The paired-difference SD is unknown before R11. Conservative bound without any pairing gain: SD ≈ 0.0043
(R6 per-block SD of Δ) gives SE ≈ 0.0012 with 12 blocks. One-sided α = 0.05 power is then ≈ 0.65 for
I = 0.0027 (pilot mean), ≈ 0.75 for I = 0.003 and ≈ 0.93 for I = 0.004. The between-block design with
8 + 8 blocks would give ≈ 0.50–0.60 for the same effects. **Power to show I ≥ δ = 0.005 as practically
meaningful is low unless the true effect is larger than in the pilot.**

## Decision rule
H1 and H2 confirmed and I ≥ δ → the paper is rebuilt around the measured interaction.
H1 (and H2) confirmed, I < δ → boundary paper with a detectable but practically small interaction.
Otherwise → boundary paper with R9–R11 reported as they are. No R12 in any case.

## Cost
Technical pilot: 36 episodes (~25 min). Campaign: 1 200 episodes; at ~30 s per episode plus restarts, ~10–11 h
(resumable, can be split over two nights).
