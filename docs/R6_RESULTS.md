# R6 — duplication-mechanism shift (2026-10-19)
Protocol: docs/r6-protocol.md (frozen before data; amendment A7: blocks 7–9 invalid, replaced by 10–12).
Data: blocks {4,5,6,10,11,12}, 50 episodes each, 10 per class; integrity check (non-reject episodes with
>50% rejected UEs) = 0 in every analysed block; 31–40 in excluded blocks 7–9. Hashes matched the protocol.
Gate: all four tasks pass. Unit of inference: held-out block (n = 6). Raw: results/r6/.

## Pre-declared criteria
| | estimate | 95% t-interval | blocks > 0 | verdict |
|---|---|---|---|---|
| P1 Brier(augmented) − Brier(la_aug), mean of 5 unseen mechanisms, h=.5 | 0.0037 | [0.0018, 0.0055] | 6/6 | **holds** |
| P2 non-inferiority on iid control (upper one-sided < 0.01) | −0.0024 (la_aug better) | upper −0.0012 | | **holds** |

## Secondary
| | estimate | 95% t | blocks > 0 |
|---|---|---|---|
| A4 (midpoint) vs augmented, unseen | −0.003 | [−0.016, 0.010] | 2/6 |
| generative BMA vs augmented, unseen | −0.044 | [−0.065, −0.024] | 0/6 |
| la_aug vs matched oracle | −0.0125 | [−0.023, −0.002] | 0/6 |
| A4-incl vs A4 | +0.006 | [−0.002, 0.014] | 4/6 |
| la_aug vs stacking (D=1) | +0.018 | [−0.002, 0.037] | 4/6 |
| decided-share error − point error (A4) | −0.009 | [−0.018, 0.000] | |
Per mechanism (la_aug gain over augmented): iid 0.0024 (6/6), burst 0.0078 (4/6, CI incl. 0), source
0.0001, incident 0.0047 (6/6), storm 0.0027 (5/6), mixed 0.0030 (6/6).
Availability curve (unseen, by h): 0: 0.0011 (n.s.); .25: 0.0034; .5: 0.0037; .75: 0.0025; 1: −0.0005.
Absolute Brier, mean over unseen, h=.5: la_aug ≈ 0.104, augmented ≈ 0.108, A4 ≈ 0.111, stacking ≈ 0.122,
presence 0.128, generative BMA ≈ 0.152, PACT ≈ 0.156, matched (oracle) 0.056–0.106.

## Honest reading (exploratory decomposition, not pre-declared)
* The P1 effect is small in absolute terms (≈ 3–4% of the Brier score) and concentrated in reg_reject
  (gain 0.012; amf/ausf/net_delay 0.0003–0.0012). Excluding reg_reject: 0.0009 [0.000, 0.0017], 5/6 blocks.
* Across all four tasks the gain is consistent under state-dependent duplication (incident: 0.0017–0.0112)
  and storms (0.0004–0.006); under source-dependent duplication there is no gain.
* Lineage helps most when part of it is recorded (h = .25–.75); with all lineage recorded the augmented
  count model is already as good, with none recorded there is nothing to use.
* The A4 midpoint rule is not competitive with learned weights; the generative model collapses under
  mechanism shift; matched training (knowing the deployment mechanism) remains clearly better than any
  method that does not know it — knowledge of the mechanism is worth more than per-message lineage.
* R6 tasks were harder than R5b (augmented AUROC 0.93 vs 0.99 for stalls), plausibly because six
  restarted blocks with 10 episodes per class vary more than three blocks with 20; not investigated.
