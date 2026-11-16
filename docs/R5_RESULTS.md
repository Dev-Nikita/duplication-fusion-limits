# R5 / R5b — block-structured campaign on the own testbed (2026-10-18)
Campaign: 3 independent blocks (restart + re-provision before each), 100 episodes per block, 20 per class
(healthy, amf_stall, ausf_stall, reg_reject, net_delay); netem available in all blocks; 300/300 episodes
complete, none excluded. Block 1 continued inside the container after the host terminal closed (no
interruption of episodes; state file and episode count consistent). Code hashes matched the frozen
protocols before the runs.

## R5 primary (docs/r5-protocol.md, unchanged) — results/r5/
Gate: all 4 tasks pass in all 3 held-out blocks (5/5 seeds) — the stronger stall configuration makes
ausf_stall learnable (R4 weak configuration stays a separate negative result).
| criterion | estimate | 95% CI (hierarchical) | per held-out block | verdict |
|---|---|---|---|---|
| C1 stacking − A4, D=8, h=.5 | 0.0248 | [0.0119, 0.0398] | 0.026 / 0.036 / 0.013 | holds |
| C2 A4 − stacking, D=1, h=.5 (NI 0.01) | 0.0002 | upper one-sided 0.0046 | | holds |
| C3 invariance h=0 | 0.0000 | | | holds |
| C4 certified error − point error | −0.033 | [−0.055, −0.015] | | **holds** (failed in R4) |
| C5 vs PACT declared | 0.069 | [0.044, 0.099] | | holds |
| C5 vs tuned log-linear pooling | 0.042 | [0.029, 0.056] | | holds |
| C5 vs geometric pooling | 0.040 | [0.031, 0.049] | | holds |
| C5 vs naive Bayes | 0.049 | [0.033, 0.066] | | holds |
| C5 vs generative BMA | −0.008 | [−0.016, −0.002] | | **fails: generative BMA better** |
Absolute Brier (mean over gated cells): D=1: stacking 0.031, A4 0.031, gen BMA 0.033;
D=8: stacking 0.073, A4 0.048, gen BMA 0.040, gen oracle 0.032, PACT-decl 0.117, TB 0.091;
D=16: stacking 0.097, A4 0.064, gen BMA 0.048. Unlike ORIGAMI and R4, the generative lineage model is
well specified here (AUROC 0.97–0.99).

## R5b secondary (docs/r5b-protocol.md, per-UE unit layer) — results/r5b/
| criterion | estimate | 95% CI | per block | verdict |
|---|---|---|---|---|
| B1 presence − A4 (8, .5) | 0.120 | [0.078, 0.161] | 0.089 / 0.127 / 0.143 | holds |
| B2 augmented − A4 (8, .5) | **−0.034** | [−0.052, −0.020] | −0.053 / −0.026 / −0.022 | **fails: augmented better** |
| B3 stacking − A4 (8, .5) | 0.088 | [0.067, 0.109] | 0.090 / 0.078 / 0.097 | holds |
| B4 NI at D=1 | 0.0001 | upper 0.002 | | holds |
| B5 invariance | 0 | | | holds |
Secondary: vs augmented at unseen D=16 −0.075 (augmented better); vs matched oracle −0.034; vs log1p
+0.056; vs PACT +0.30; vs pooling +0.18; vs naive Bayes +0.20; decided-share error − point error −0.056.
Absolute Brier (8, .5): augmented 0.036, matched 0.036, A4 0.069, log1p 0.125, stacking 0.158,
presence 0.189, TB 0.254, naive 0.273, PACT-decl 0.368.

## Exploratory (post hoc on R5, after B2 failed) — lineage-aware augmented stacking
`experiments/r5b/explore_lineage_augmented.py`: recorded units + per-slot count of unrecorded copies,
trained with the same augmentation. vs augmented (Brier difference, positive = lineage helps):
D=1: +0.0002; D=8,h=.5: +0.0004; D=8,h=1: −0.003; **D=16 (unseen), h=.5: +0.005** (all 3 blocks > 0);
D=16, h=1: −0.019.

## What this means
1. On the R5 layer the pre-declared claim holds in full (C1–C4, C5 except generative BMA).
2. Against a lineage-free model trained across duplication regimes, the A4 point estimate loses clearly.
   Its midpoint rule (half a unit per unrecorded copy) is the weak part, not the use of lineage.
3. With augmentation, recorded lineage adds value only beyond the training range of D, and only when
   most lineage is recorded. Caveat: the augmentation used exactly the test-time duplication mechanism —
   oracle knowledge of how copies arise. Whether lineage pays off when the duplication mechanism at
   deployment differs from the one simulated in training is the open, decisive question (R6).
