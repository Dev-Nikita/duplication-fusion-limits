# G3 results (2026-10-17) — decision-critical lineage queries
Protocol docs/g3-protocol.md (declared before the run); theory docs/theory/G3_FORMALIZATION.md;
raw results/g3/pilot.json. 30 repetitions × 400 test episodes per hide level; starting space = G2
calibrated sets (δ = 0.05); unit query cost; population = episodes undecided before any query
(1,621 at hide 0.5; 2,434 at hide 0.75).

| Policy | hide 0.5: queries | hide 0.75: queries | decided | final decision ≠ oracle-posterior decision |
|---|---|---|---|---|
| random | 1.70 | 2.15 | 0.83 | 0.3–0.6% |
| max_uncertainty | 1.71 | 2.17 | 0.83 | |
| expected_interval_reduction | 1.56 | 1.95 | 0.83 | |
| decision_robustness (SDP-style) | **1.44** | **1.75** | 0.84 | |
| ec2 (DRD) | 1.44 | 1.75 | 0.84 | |
| exact optimum (DP, expected) | 1.47 | 1.75 | — | — |

* Q1 **holds**: decision-robustness and EC2 need 0.26 (hide 0.5) and 0.40 (hide 0.75) fewer queries
  than random / max-uncertainty, 95% CIs exclude 0 (≈15–19% fewer).
* Q2: both are at the exact optimum within noise (realised 1.43 vs expected optimum 1.47 on the DP
  subset at hide 0.5; 1.72 vs 1.75 at 0.75). **There is no room for a better policy** in this regime.
* Q3: the single-unresolved-per-cluster regime of Prop. G3.1 covers only 24% / 8% of undecided
  episodes in this generator (candidate universes merge clusters).
* ~17% of episodes stay undecided after all queries: the residual width is parameter uncertainty
  (FRESH box), which lineage queries cannot remove.
* Certificate failures (truth outside calibrated set) 2–3%, consistent with δ = 0.05.
* Final robust decisions agree with the oracle-posterior decision in 99.4–99.7% of decided episodes.

**Verdict (pre-declared gate): no methods contribution from G3.** What G3 contributes: Prop. G3.1
(reduction to LTF-SBFE — transfers known hardness/approximation results to provenance resolution),
an empirical demonstration that decision-focused queries save 15–19% of provenance look-ups at no loss
of certified accuracy, and the finding that remaining indecision is parameter-, not lineage-driven.
