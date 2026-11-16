# R6-X — ablations and decomposition (secondary, 2026-10-19)
Protocol docs/r6x-protocol.md (frozen before fitting). Reproduction check: augmented − la_aug = 0.0037
[0.0018, 0.0055], 6/6 (identical to R6 P1).
| criterion | estimate | 95% t | blocks > 0 | verdict |
|---|---|---|---|---|
| A1 placebo − la_aug | 0.0079 | [−0.0016, 0.0175] | 4/6 | **fails** (worse for every fault: 0.004–0.018) |
| A2 capacity-matched − la_aug | 0.0084 | [0.0015, 0.0154] | 5/6 | holds |
| A3 tuned augmented − tuned la_aug | 0.0037 | [0.0017, 0.0056] | 6/6 | holds |
Excluding reg_reject (descriptive): A2 0.0068 [−0.0023, 0.0159]; A3 0.0003 [−0.0004, 0.0011].
Consequence (pre-declared): the R6 gain is described as a lineage-separated *representation* effect,
consistent with but not established as an effect of correct lineage.

Decomposition (descriptive, Prop. 2 of the paper): mismatch = Brier(augmented) − Brier(matched) is large
only under 'incident' (0.054–0.103 for stalls/delay) and moderate under 'mixed' (0.015–0.021). Across 24
task×mechanism cells the la_aug gain rises with the mismatch (Spearman 0.48, p = 0.019) but recovers ≈ 22%
of it. For reg_reject la_aug gains where the mismatch is nil (refinement: distinct rejected UEs vs messages).
