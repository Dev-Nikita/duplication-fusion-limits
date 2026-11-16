# Amendments log

## A1 — 2026-10-17 (exploratory; added AFTER the P1–P4 results were seen)
Reason: two questions a hostile reviewer would ask that the declared protocol did not cover.
1. P1 measured |Δp| without conditioning on the sign of the copied evidence, so H1 as declared
   ("lower bound DS_k = 0") is **not supported as declared**: mixed-sign copies move the lower bound.
   A1a re-analyses the same P1 stream split by sign.
2. The generator never produced truly independent unresolved observations, so `paef_bma`'s
   uniform lineage prior was close to the generator by construction. A1b adds
   `p_true_fresh` (share of unresolved secondaries that really are independent origins).
3. A1c re-runs P2 hide=0.5 with the amended generator to show the earlier numbers are reproduced
   bit-for-bit (`p_true_fresh = 0` consumes no extra random draws).

A1 results are exploratory and are labelled as such wherever they are used.

## A2 — 2026-10-17 (leakage audit; exploratory; requested by external review)
* A2a static: only `oracle` reads ground-truth fields (`baselines/simple.py:37`). Pass.
* Found by inspection: the generator gave every observation `fresh_theta` = parameters of its TRUE
  origin, and baselines' marginals were computed from the true origin. Both are truth-derived.
* A2b: replaced by deployable quantities (`simulation/agnostic.py`). Brier / risk changes ≤ 0.005 for
  every method; rankings unchanged. **But coverage of the oracle posterior drops from 1.000 to 0.730 at
  p_true_fresh = 1**: the earlier 100% depended on the independent-origin hypothesis carrying the true
  parameters. This confirms the reviewer's Problem 1.
* A2c: a credal box for independent-origin parameters (min/max over declared origins) restores
  coverage to 1.000 in all cells, at a cost: decided share 0.78 → 0.73, mean width 0.23 → 0.29
  (hide 0.75). Still a by-construction guarantee (the box contains the truth); a data-calibrated
  candidate set with P ≥ 1−δ is the open task (G2b).
* Selective risk: the interval still loses to `paef_bma` at equal coverage in every A2c cell
  (e.g. 0.088 vs 0.076). K1 stands.

## A3 — 2026-10-17 (exploratory; added after seeing G2-P)
Observation: in G2-P posterior coverage (~0.99) far exceeded lineage coverage (~0.95), so lineage-level
calibration is conservative for the reported object. A3 calibrates the envelope itself with nested
conformal over the lineage threshold q (`experiments/g2/run_a3_posterior_targeted.py`, same episode
streams as G2-P). Guarantee is for the true-model posterior (object 2), not for the lineage (object 1).
Not new as a method (novelty-audit H1–H3).

## A4 — 2026-10-17 (exploratory; added after seeing the R2 results)
R2 showed (i) the generative lineage model is misspecified on real data (ECE 0.106 > 0.05: R2-H3
triggered), (ii) lineage-blind stacking is far better in distribution (Brier 0.128 vs 0.182) but degrades
under duplication shift (+0.067 from D = 1 to 8, CI [0.054, 0.082]). A4 combines both: a logistic model
trained on countable units (lineage known on training runs), with known copies dropped at test time and
hidden copies carried as an interval (certificate) or with a uniform prior (point). Because it was
designed after seeing R2, its R2 numbers are exploratory; confirmation is R3 (docs/r3-protocol.md),
frozen before any R3 data were processed.

## A5 (2026-10-18, before any R5 episode existed) — campaign.py repair
The edit that fixed the R5 intensity plan on resume accidentally removed the end of `campaign()` and the
header of `r5_block()`; three launch attempts on the Mac stopped after preflight with
`NameError: r5_block` and produced **no episodes** (episode count stayed 128). `campaign()` was restored
verbatim from the R4 version and `r5_block()` re-inserted with the same schedule, intensities and seeds;
tested end to end in a container (4 episodes, netem fallback). Frozen hash updated in docs/r5-protocol.md
(d923970b8ca7584e5a4357135c876a0b4e9777d9c647dbbfc1853163e1d8a184). No analysis code changed.

## A6 (2026-10-18) — doc wording only
PACT meet is the component-wise minimum (lattice meet); the code always used np.min. The words
"component-wise max" in r5-protocol.md and in the paper were wrong and are corrected. No code changed.

## A7 (2026-10-18) — R6 blocks 7–9 invalid; replaced by blocks 10–12
Cause (my bug): episode IMSIs are base + 200 + 20·idx, but each block provisioned only 12 000
subscribers. From episode index 590 onward every UE was unprovisioned, so every episode in blocks 8–9
and 38 of 50 in block 7 looks like a registration-reject episode regardless of its label.
Detection: the learnability gate failed for every task (calibration AUROC ≈ 0.47, including the trivially
separable reg_reject); a data check then showed ue_rej ≈ 1.0 for non-reject episodes with idx ≥ 590.
No R6 outcome metric was examined (all criteria evaluated to NaN because no task passed the gate).
R5 (max idx 427) and R4 are unaffected. Fix: run.sh provisions 25 000 subscribers and aborts if a block
would exceed the provisioned range; blocks 10–12 are run with the same configuration and seeds
20261025 + 1000·b. Analysis blocks: {4, 5, 6, 10, 11, 12}; all episodes of blocks 7–9 are excluded
(block 7 entirely, to keep blocks complete). run_r6.py changed only in the BLOCKS constant.

## A8 (2026-10-24, after R12) — normal empty-gate termination
The frozen R12 evaluator recorded that no system passed the pre-declared learnability gate and then raised
`KeyError` while trying to pivot the empty analysis set. The evaluator now returns the normal terminal
status `NOT_TESTABLE`, sets H-W, H-WB, H-A and H-C to `null`, and sets `GO` to `false`. No feature,
prediction, AUROC, Brier value, threshold, hypothesis or decision rule changed. A regression test covers
the empty-gate path. The frozen evaluator hash remains recorded in `docs/r12-protocol.md`; the corrected
post-outcome evaluator hash is
`af0d5b1fac6c45f434cd127ba06d880b30120c2a22b6af806fd5d06c58282290`.
