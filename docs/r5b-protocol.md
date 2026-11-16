# R5b protocol — per-UE unit layer and lineage-free baselines on the R5 campaign
**Frozen 2026-10-18, after the R5 campaign finished and before any R5 or R5b metric was computed.**
Design inputs: the R4 raw episodes only (ep_0008–0127) and the post-hoc R4 finding that a lineage-free
presence baseline beats A4 in the R4 observation layer (docs/R4_RESULTS.md, post-hoc section).

## Status (stated in the paper exactly like this)
R5b is a **secondary, prospectively specified analysis** of the R5 data. The observation layer was
designed after the presence finding on R4, so R5b is not an independent confirmation; an independent
campaign (R6) with this layer frozen is required for a confirmatory claim. The R5 primary analysis
(docs/r5-protocol.md) is run and reported unchanged.

## Why a new layer
In the R4/R5 layer each slot holds one reading and every copy is derived, so binarising counts removes
duplication without lineage. A4 can only matter where the number of *independent* units in a slot is
informative. In a core network the natural units are subscribers: several UEs failing independently is
stronger evidence than one, while monitoring re-emissions of one UE's alarm are not.

## Unit layer (`experiments/r5b/build_units.py`, `src/paef/real/units.py`)
* Origins = UEs of an episode (19–20). Four readings per UE from two natural channels:
  ue_slow (UE-log registration latency > p90 of healthy training UEs, or no success; declared primary),
  amf_slow (AMF-log SUCI→Registration-complete latency > p90, or none), ue_rej, amf_rej.
* Slots = reading types shared by all UEs; counts range over affected UEs.
* Copies: each firing per-UE alarm re-emitted D times; lineage of secondary readings and copies hidden
  with probability h (as in R5).

## Methods
A4 (unit features, exact interval, midpoint) and: stacking on counts (trained at D=1); **presence**
(any alarm of a type); **log1p** counts; **augmented** stacking (trained on D ∈ {0,1,2,4,8}, h ∈ grid,
5 replicas — D=16 unseen); **matched** stacking (trained at the test D and h — oracle knowledge of the
deployment duplication, reference); naive Bayes, tuned log-linear pooling, PACT-style meet (declared;
true partition as reference) with one generative model shared across UEs.

## Evaluation
As R5: leave-one-block-out, 5 seeds, 70/30 calibration split, learnability gate (stacking AUROC ≥ 0.65 on
calibration; fold kept if ≥ 3/5 seeds pass; task kept if ≥ 2/3 folds), unit of resampling = (task,
held-out block), hierarchical bootstrap (20 000).

## Criteria (A4 is said to add value over lineage-free fusion only if B1 and B2 hold)
* B1 Brier(presence) − Brier(A4) at (8, 0.5): CI > 0 and positive in every held-out block.
* B2 Brier(augmented) − Brier(A4) at (8, 0.5): CI > 0 and positive in every held-out block.
* B3 Brier(stacking) − Brier(A4) at (8, 0.5): as C1.
* B4 non-inferiority at (1, 0.5): one-sided 95% upper bound of Brier(A4) − Brier(stacking) < 0.01.
* B5 invariance |Brier_A4(16,0) − Brier_A4(0,0)| < 0.005.
* Secondary (reported, no decision): A4 vs augmented at unseen D = 16; vs matched oracle; vs log1p,
  PACT, pooling, naive Bayes; vs presence at D = 1; decided-share error; absolute Brier table.
Three blocks give limited between-block information; results are reported as reproducibility across
three blocks, not as a strong statistical guarantee.

## Frozen code (sha256)
```
dc6a7aba1311ba4410e2e4b65d357b1e5d96dfd5286bc5226eb93dd9ee0f9531  experiments/r5b/build_units.py
d6b51f8b9abc10749d0c8b46320cb7b8cea781fb310b7c68da8e6bbeae1a9c38  experiments/r5b/run_r5b.py
66e0df2ed61b824849a614efdde8a5f5c4864d1436759a97db480af666934861  experiments/r5b/evaluate_r5b.py
8ae05c4bdcba2b8eab42cc5ec722d1ec42987e6047340ca8ea72cb9a2792a0eb  src/paef/real/units.py
2d61611e06f6d615e4d43d6b0f88ec512e8e411fa13ba550de115a4bee4a6c3d  src/paef/real/generic.py
```
