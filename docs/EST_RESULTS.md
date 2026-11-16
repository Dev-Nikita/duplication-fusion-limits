# Estimation-error analysis — results (protocol: docs/est-protocol.md, frozen before running)

Output: `results/est/estimation_error.csv`, `results/est/summary_by_D.csv`. 108 folds × 5 caps.

Mean exact worst-case Brier on the held-out block:

| D | plugin | presence | smoothed | tvrobust | select |
|---|---|---|---|---|---|
| 1 | 0.098 | 0.170 | 0.099 | 0.138 | 0.097 |
| 2 | 0.111 | 0.170 | 0.112 | 0.151 | 0.111 |
| 4 | 0.126 | 0.170 | 0.128 | 0.166 | 0.127 |
| 8 | 0.145 | 0.170 | 0.147 | 0.185 | 0.145 |
| 16 | 0.164 | 0.170 | 0.164 | 0.202 | 0.163 |

Pre-written expectations:
- **E1 held.** The select rule's mean regret against the per-fold better of {plugin, presence} is 0.0021–0.0041 (≤ 0.01).
- **E2 failed.** tvrobust beats plugin in only 19/108 folds at D=16. The block-variability radius ε ≈ 0.30 (0.20–0.43) is too conservative.
- **E3 held.** |smoothed − plugin| in mean W is ≤ 0.0015 for D ≤ 4.

Wins over presence (of 108 folds) at D = 1, 2, 4, 8, 16:

| Rule | D=1 | D=2 | D=4 | D=8 | D=16 |
|---|---|---|---|---|---|
| plugin | 80 | 78 | 76 | 66 | 47 |
| smoothed | 89 | 83 | 79 | 74 | 61 |
| select | 73 | 71 | 59 | 46 | 27 |

The select rule chose plugin in 81% of folds at D=1 and 36% at D=16. By dataset at D=16, presence is better than
plugin by ≤ 0.005 in R10, R6 and R11-loaded; plugin is better by 0.025 in R11-control.

Unplanned observation, reported as exploratory: smoothed wins against presence more often than plugin (61 vs 47
at D=16), although its mean W is almost the same. It was not a pre-written hypothesis and is not used in the paper.
