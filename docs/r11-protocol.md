# R11 protocol — Host contention × evidence dependence
**Frozen 2026-10-22, before any R11 confirmatory episode exists** (the draft was docs/r11-protocol-DRAFT.md; the
only changes are W, the pilot record, the analysis details and the hashes). R11 is the **last** campaign: if H1
is not confirmed, no further campaign is run and the boundary paper is submitted with R9, R10 and R11.

## History that must be disclosed
R10 (confirmatory) did not confirm P1. A post-hoc pilot of the frozen R10 code on R6 data (exploratory, already
inspected) showed a positive LD gap where healthy-episode latency was noisy, and none in R10. R11 asks whether
this is reproduced when the disturbance is manipulated experimentally. R6, R10 and the pilot are not reused.

## Manipulation
Load sidecar (testbed/scripts/loadgen.py) on the same Docker VM: W busy workers (CPU spin + 64 MB memory
traffic) during an episode (warm-up 4 s), 0 otherwise. W is fixed by the technical pilot rule in
experiments/r11/pilot_check.py (manipulation check on healthy-episode latency only; no outcome model is fitted).

**Technical pilot (2026-10-22, ep_1178–ep_1213, blocks 900–905, order L0 L1 L2 L2 L1 L0; not used for any test):**

| W (busy workers) | VM CPU busy | healthy median latency, median / q90 (s) | stall median (s) | NFs alive | max reject share |
|---|---|---|---|---|---|
| 0 | 0.10 | 0.161 / 0.180 | 0.669 | all | 0 |
| 5 | 0.60 | 0.194 / 0.352 | 1.143 | all | 0 |
| 10 | 0.99 | 1.052 / 1.427 | 2.359 | all | 0 |

Rule outcome: **GO, W = 10** (only level with healthy q90 ≥ 0.87 s; R6 blocks 10–12: 0.87–2.0 s). The loaded
condition is therefore a saturated host (Docker VM CPU ≈ 100 %), and this is how it is described. Healthy latency
returned to baseline in the L0 sub-run that followed L2 (no carry-over). Unloaded latency differs from R10
(R10 healthy ≈ 0.42 s), which is uncontrolled host state; the paired within-block design is meant to absorb this.
Measured duration: 32.5 s per episode.

## Design (paired within block; chosen for power, see below)
* Blocks 21–32 (n = 12), core restart and fresh subscriber database (25 000) before each block, as in R10.
* Per block: 10 episodes per class (healthy, amf_stall, ausf_stall, net_delay, reg_reject) per condition
  (control / loaded) = 100 episodes. The order of (class, condition) is a random permutation fixed in advance
  from the block seed, so the condition is independent of the class and balanced in time.
* The condition is recorded in meta.json only. It is not a feature, and features contain no time stamps,
  block ids or episode indices.

## Analysis (experiments/r11/add_cond.py → run_r11.py → evaluate_r11.py; R10 pipeline otherwise unchanged)
* Leave-one-block-out. Models are trained and calibrated **within condition** on the other 11 blocks and
  tested on the held-out block's episodes of the same condition. Train and test never share an episode or
  copies derived from one. 5 Monte-Carlo seeds per (task, block, condition).
* Methods: logistic on counts (augmented), GBDT on counts (gbdt_aug), GBDT on presence features
  (type-level deduplication baseline, gbdt_dedup), GBDT on lineage-separated features (gbdt_la), GBDT on exact lineage
  (gbdt_unit), and a GBDT trained on both LD mechanisms (gbdt_mix_oracle). Mechanisms: none, LI set, LD pair.
* reg_reject is not analysed (trivially separable; R6–R10). Calibration AUROC is reported, not used as a gate.
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

## Frozen code (sha256)
```
96a1e6cfdc6429ea4865bcd20de38d35fe8152a5e1a243b0873c85b141246030  testbed/run_r11.sh
f9732e256e0a2fe0ad058abd74d782127898505d6d7df2aedb7beb3914a8ef23  testbed/docker-compose.r11.yml
6032e26ef6980198515d6cd0633974095a9184386838a9a959bbe5140bcc1547  testbed/scripts/loadgen.py
a64d0647820aba4aa7cdd49396a594fd8112cfe9a8e44d41418c5be1b7bd26ee  testbed/scripts/campaign_r11.py
d923970b8ca7584e5a4357135c876a0b4e9777d9c647dbbfc1853163e1d8a184  testbed/scripts/campaign.py
944f0dc1d6eb670798e99afa2c2fc74248f0e2345189375b8afca385dabb8844  testbed/scripts/provision.js
f757891d775d58645a6d07671d6a858be7faa29836e9d265e8af9856178af4d9  experiments/r11/add_cond.py
5f8bd1d3887dc70fcab340abd126d5b481fd7218e4abbf718d55b230ffa0f563  experiments/r11/run_r11.py
a92e2221b828321c5aab79cccf89989189df9be953cb3301b7ebec05f5e30d47  experiments/r11/evaluate_r11.py
c634187a45b707eb1af00bad385fc410ef238e6abb16a94cab20530e96513b04  experiments/r11/pilot_check.py
3f9f5fa8c9a973ef436f2e3134ff6e3fef8bee146bc221710c0281158d2f08f7  experiments/r10/run_r10.py
05f820fff0f580692d31ab0a66c64e7d10728b70e10b5650bbe464bf22a247f8  src/paef/real/mechanisms_r10.py
f36b09be737d9cf1408674acb4ea7b234a2cdfb2ce88282f26bf6700d1b57aec  src/paef/real/mechanisms.py
8ae05c4bdcba2b8eab42cc5ec722d1ec42987e6047340ca8ea72cb9a2792a0eb  src/paef/real/units.py
dc6a7aba1311ba4410e2e4b65d357b1e5d96dfd5286bc5226eb93dd9ee0f9531  experiments/r5b/build_units.py
d6b51f8b9abc10749d0c8b46320cb7b8cea781fb310b7c68da8e6bbeae1a9c38  experiments/r5b/run_r5b.py
```

## Commands
```
./run_r11.sh r11all 10          # on the Mac, from testbed/ (resumable)
python3 -I experiments/r5b/build_units.py <root> results/r11/units.csv
python3 -I experiments/r11/add_cond.py <root> results/r11/units.csv results/r11/units_cond.csv
python3 -W ignore -I experiments/r11/run_r11.py results/r11/units_cond.csv results/r11
python3 -W ignore -I experiments/r11/evaluate_r11.py results/r11/r11_cells.csv results/r11/r11_gate.csv results/r11/r11_results.json
```
