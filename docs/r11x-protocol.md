# R11-X protocol — observation channel × evidence lineage (confirmatory replication of a post-hoc finding)
**Frozen 2026-10-23, after R11 and before running any alarm-only analysis on R10 or R6 data.**

## History (must be disclosed in the paper)
R11 (confirmatory) did not confirm H1. While interpreting R11 we noticed a property of the feature maps used
in R5b–R11 (and of the Alertmanager emulation, R8). For every reading type the "lineage-free" count features
contain two counts: messages reporting a firing reading, and readings reporting not firing. Duplication only
ever adds firing messages. With a known number of reporting units per window (19–20 UEs), the non-firing
count reveals the number of firing **units** exactly. The lineage-free learners therefore had access to the
unit count through the non-firing reports, the *denominator*.

Exploratory check on R11 data (already unblinded; experiments/r11x/alarm_only.py, results/r11x/): with the
firing column only, the worst-case LD loss of gbdt_aug relative to exact lineage is +0.159 Brier (control,
12/12 blocks > 0.005) and +0.038 (loaded, 10/12). Lineage-separated features (gbdt_la) stay within
+0.002/+0.003, and with full counts the loss is +0.002/+0.003. This is a post-hoc finding and is reported as such.

## Confirmatory replication (data never analysed with the alarm-only channel)
* R10: blocks 13–18 (results/r10/units.csv).
* R6: blocks 4, 5, 6, 10, 11, 12 (results/r6/units.csv).
Each dataset is analysed separately with experiments/r11x/replicate.py (frozen R11 pipeline, single
condition, 3 seeds, leave-one-block-out), once with channel = alarm and once with channel = full, and
evaluated with experiments/r11x/evaluate_r11x.py.

## Predictions (δ = 0.005, the R10 negligibility margin; unit = held-out block)
* **X1:** without the denominator, lineage matters. G_alarm(gbdt_aug) has one-sided 95 % lower bound > δ,
  and ≥ 5/6 of blocks are > δ.
* **X2:** lineage-separated features recover it. G_alarm(gbdt_la) has one-sided 95 % upper bound < δ.
* **X3:** with the denominator, lineage is redundant. G_full(gbdt_aug) has one-sided 95 % upper bound < δ.
  Not blind: the full-channel LD gaps of these data are partly known (R10 results; R6 pilot). Reported, but
  it carries no confirmatory weight.

## Decision rule
X1 and X2 hold in **both** datasets → the paper is rebuilt around the measured boundary: evidence lineage is
redundant when the channel carries negative (non-firing) reports from known units, and decisive when it
carries firing alerts only. Otherwise → boundary paper with R9–R11 and this analysis reported as they are.
No new testbed campaign in either case.

## Secondary (reported, no test)
R8 Alertmanager emulation re-analysed with the firing-only channel. The R9 capacity result re-read in light
of the denominator.

## Frozen code (sha256)
```
7d0c8f821e7f12d59d50d0874a32b164813884c4708de71a97d266df253ce1e9  experiments/r11x/replicate.py
2038481fc3a9717ebbeb572f0c4ff16f5bcd0bfb8855d0322c5dd5679fe999a9  experiments/r11x/evaluate_r11x.py
5f8bd1d3887dc70fcab340abd126d5b481fd7218e4abbf718d55b230ffa0f563  experiments/r11/run_r11.py
3f9f5fa8c9a973ef436f2e3134ff6e3fef8bee146bc221710c0281158d2f08f7  experiments/r10/run_r10.py
d6b51f8b9abc10749d0c8b46320cb7b8cea781fb310b7c68da8e6bbeae1a9c38  experiments/r5b/run_r5b.py
8ae05c4bdcba2b8eab42cc5ec722d1ec42987e6047340ca8ea72cb9a2792a0eb  src/paef/real/units.py
f36b09be737d9cf1408674acb4ea7b234a2cdfb2ce88282f26bf6700d1b57aec  src/paef/real/mechanisms.py
05f820fff0f580692d31ab0a66c64e7d10728b70e10b5650bbe464bf22a247f8  src/paef/real/mechanisms_r10.py
1fb807bfe706996ca0ab5068a1d9e75f7662af79dd4964213113ce7982f346a5  results/r10/units.csv
84795685f867a2e9e037dc16f7d2ed52914b8fa2e81beb8c2222b6ba40511187  results/r6/units.csv
```
## Commands
```
for D in "r10 results/r10/units.csv 13,14,15,16,17,18" "r6 results/r6/units.csv 4,5,6,10,11,12"; do set -- $D
  for ch in alarm full; do python3 -W ignore -I experiments/r11x/replicate.py $2 results/r11x_rep/$1_$ch $ch $3; done
  python3 -W ignore -I experiments/r11x/evaluate_r11x.py results/r11x_rep/$1_alarm/r11_cells.csv results/r11x_rep/$1_full/r11_cells.csv results/r11x_rep/$1.json
done
```

## Addendum A (2026-10-24, before any R11-X run on R10 or R6) — terminology and exploratory checks
* **Terminology:** R11-X is a *retrospective, protocol-frozen cross-campaign validation*. R10 and R6 are not
  independent new samples: both shaped the research programme, R10 directly. X1–X3 and their criteria are
  unchanged.
* **Code check of U_alarm = M − N_nonfiring (per reading type, per window).** It holds exactly when each unit
  delivers exactly one reading of each type per window, only firing readings are duplicated, no non-firing
  report is lost, and M is known. In this testbed all four hold. One reading per UE per type is built in
  episodes(): an unseen AMF reading is coded, not dropped. Copies attach only to readings = 1 (make_window,
  make_window_r10). Nothing removes non-firing reports. M = 20 in 296/300 R10 episodes and 19 in 4.
  The equality breaks with lost non-firing reports, an unknown or varying population, duplicated non-firing
  reports, or several readings per unit per window.
* **Information available to each model** (same training windows, same augmentation, same calibration split):
  gbdt_unit gets K = 4 distinct-firing-unit counts. gbdt_aug and logistic get the channel features.
  gbdt_la gets distinct firing units, recorded-copy counts and unrecorded-copy counts (3K), with no other
  extra inputs. gbdt_mix_oracle gets the channel features but is trained on labelled LD windows. No model sees
  block ids, time, episode ids or the load condition.
* **Exploratory, not confirmatory** (experiments/r11x/explore_x.py; same R10 and R6 blocks; Brier and log loss
  per mechanism and per block; GBDT and logistic). Reference everywhere: gbdt_unit.
  * X4: firing counts + oracle distinct-firing-unit counts. Does lineage add anything beyond multiplicity?
  * X5: firing counts + population size M, no non-firing reports. Is knowing M enough?
  * X6: full channel with 30 % of non-firing reports lost at random.
  * X7: full channel with an unknown random population per window, M ~ U{8..20}.
* After R11-X: one consolidated report R4–R11-X with the history of all hypotheses, results and changes.
  No new testbed campaign.
```
ffbf7361c18e192746fe04c101cb0fdce3e89abf0083a5bcddac8316999dad87  experiments/r11x/explore_x.py
```
