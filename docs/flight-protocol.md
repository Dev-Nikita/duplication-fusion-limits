# External study on naturally copied evidence — Flight data set (protocol frozen 2026-11-15T17:52:29Z, before any outcome was computed)

## Why this data set
The fixed criteria for an external check were: (a) real, naturally arising copies of reports, not imposed by us;
(b) ground-truth labels; (c) source identifiers that allow a lineage proxy; (d) a public data set from a different
domain than the 5G testbed. The Flight data set of Li et al. (VLDB 2012; lunadong.com/fusionDataSets.htm) meets them:
31 web sources report on about 1,200 flights per day for December 2011; many sources copy from each other; a gold
standard exists for 2,709 (day, flight) items. sha256 of the archives: data_external/lunadong_fusion/SHA256SUMS.

## What was already known (honest status)
On 2026-10-24 a feasibility check (docs/FEASIBILITY_EXTERNAL_DATA_2026-10-24.md) computed, for the attribute "actual
departure time", the accuracy of majority voting over sources and over agreement clusters, and the cluster sizes.
Today's feasibility check computed only the label base rate (0.214), the source coverage, the airline counts, and — with
**permuted labels** — that the pipeline runs. No Brier score, contrast or descriptor of this protocol had been computed
on the real labels when this file was written. The study is therefore **pre-specified but not blind to the data set**; it
is reported as an external illustration with a frozen protocol, not as an independent replication.

## Task and channels
* Item: gold (day, flight) with scheduled and actual departure. y = 1 if gold actual departure is ≥ 15 min after the gold
  schedule. The schedule is treated as published metadata.
* A source **fires** on an item if its reported actual departure is ≥ 15 min after the schedule; it sends a **negative
  report** if it reports an earlier actual departure; otherwise it is **silent** (population is not fixed: coverage varies).
* Lineage proxy: copy clusters = single linkage over source pairs with ≥ 50 co-reported items and agreement within 1 min
  ≥ τ, built on training items only, without the truth. Primary τ = 0.95; sensitivity τ = 0.90 and 0.99.
  This is a proxy, not ground-truth provenance.
* Channels: A (firing sources, copies included), U (firing clusters), P = 1{A>0}, N (negative-reporting sources),
  NU (reporting clusters that do not fire).

## Rules (fitted on training items only; scored by Brier on held-out items under the natural copying)
count (GBDT on A), unique (GBDT on U), presence (GBDT on P), count_neg (GBDT on A, N), unique_neg (GBDT on U, NU);
GBDT = histogram gradient boosting, depth 3, 200 iterations, learning rate 0.1 (the paper's fixed setting).
naive = table P̂(y | U) applied to the raw count A; minimax = plug-in finite-cap minimax rule of the training law
p(y, U) with cap D̂ = (largest training cluster) − 1, applied to A.

## Splits
* week: leave one of four date blocks out (days 1–7, 8–14, 15–21, 22–31). Same sources, same copying: stable mechanism.
* airline: leave one airline out (AA, UA, CO). Airline sites and coverage differ by airline, so the copy multiplicity of
  a firing cluster changes naturally between training and test: natural mechanism shift.

## Hypotheses (each is reported whether it holds or not)
* **H-F1 (primary).** Under the airline split, the count rule is worse than the unique rule:
  Brier(count) − Brier(unique) ≥ 0.005 and the 90 % day-cluster bootstrap lower bound is > 0 (primary τ).
* **H-F2.** Under the week split (stable mechanism) the same difference is within ±0.005.
* **H-F3.** Under the airline split, the count rule with negative reports is not worse than the unique rule by more than
  0.005 (negatives substitute for lineage; population not fixed, so Proposition 1 does not apply exactly).
* **Descriptors (no hypothesis).** On all items: R_B(U), R_B(P), R_B(A) under the natural copying, the finite-cap
  minimax value at D̂, the count-only value, the nature index (R_B(A) − R_B(U)) / (R*_D̂ − R_B(U)) (0 = copies are
  harmless, 1 = nature is as bad as the worst case, < 0 = copies are informative), D_sat, δ_D, and copies per firing
  cluster by class.

## Fixed in advance
Delay threshold 15 min; tolerance 1 min; minimum co-reports 50; τ grid; GBDT hyperparameters and seed; bootstrap
(2,000 day resamples, 5–95 %). The day-cluster bootstrap is descriptive: days are few (31) and the airline split
reuses days across folds. No change of threshold, τ, split, rule family or data set after seeing results. A null or
negative outcome is reported as such, and practical claims are revised accordingly.

## Code (sha256)
```
b6b5a14a0b0993522cb2c32e7d91d5826857d7ccacaaa1fa300b185d16f5564e  experiments/flight/flight_study.py
c43c3d192305d0deeae696d8113a6174771ef4919bf4a930513b87936da9a4ee  experiments/flight/flight_evaluate.py
c11f989b4cb406df65f93bca6cf01997a7e314f7dd9fa0e69535af2474646857  experiments/flight/flight_io.py
3517932d8767a38bb0de45bd160c4c450a95a70fe3bdb6953303418b7552a1d9  tools/finite_cap_cvx.py
f6b9fbd5c9bb6837232407651e1fbecbcfef80ac02407817e1030faabfeee98b  tools/finite_cap_solvers.py
df0a4e36fc1f9d96e130649881f9f5c40bde125dc11784ac16f8aa4d1901f533  tools/saturation_closed_form.py
01f93630e79c0f7ad27257219b2c15753591a70ae49501ce7a5c1b3acd039306  tools/quantitative_saturation.py
```
