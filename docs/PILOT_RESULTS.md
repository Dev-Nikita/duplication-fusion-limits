# Pilot results and go/no-go (2026-10-17)

Raw output: `results/pilot/pilot_report.json`, `results/pilot/amendment_a1.json`.
Synthetic only; n = 3000 episodes per cell; seed 20261017. Not paper results.

## Against the declared hypotheses (docs/pilot-protocol.md)
| ID | Declared | Outcome |
|----|----------|---------|
| S0 | oracle posterior inside [lower, upper] when metadata is correct | **Holds**: 100% in every cell (also under every A1b lineage prior) |
| H1 | known copies: DS_k = 0 for lineage-aware rules; unknown copies: lower bound DS_k = 0 | Known copies: **holds**. Unknown copies: **not as declared** (mixed signs). A1a: the bound *opposite* to the copy's evidence is exactly invariant (max |Δ| ≤ 2e-16 for k = 1…8, both candidate regimes); the other bound widens |
| H2 | interval rule beats every non-oracle point rule at equal coverage (hide ≥ 0.5) | **Fails (K1 triggered)**. Interval beats naive/declared/PACT/cautious, but ranking by `paef_bma` confidence has lower risk at the same coverage in every cell (e.g. hide 0.5: 0.105 vs 0.095) |
| H3 | `paef_bma` Brier below declared_only and both PACT variants (hide ≥ 0.5) | **Holds** under the declared generator (hide 0.5: 0.105 vs 0.120 / 0.125 / 0.175). **A1b: depends on the lineage prior** — at p_true_fresh = 1 declared_only equals the oracle (0.0675) and bma is worse (0.0778) |
| H4 | interval decides ≥ 50% at hide = 0.5 | **Holds**: 83.5% |
| H5 | resolving the max-value link decides more episodes than a random link (CIs disjoint) | **Not met**: 34.0% [28.7, 39.3] vs 24.7% [20.0, 29.7], CIs overlap. Width reduction is larger (1.11 vs 0.83 log-odds, CIs disjoint) but that was not the declared metric |

## What this means for the paper
1. **The interval is a certificate, not a decision rule.** Its value is the guarantee (S0, and the
   one-sided invariance of A1a, which is a provable proposition), not selective accuracy. The
   paper must not claim the interval abstains better than a calibrated point estimate.
2. **The point estimate needs a learned lineage prior.** Uniform model averaging wins only when its
   prior roughly matches reality. Next step: estimate P(shared origin | source types, timing) from the
   subset of events whose lineage is recoverable (5G procedure / message IDs) and test transfer.
3. **PACT-style meet with conservative closure is badly over-conservative** (AUROC 0.73–0.82): merging
   everything that *might* share an origin destroys independent corroboration. PACT with the declared
   partition degrades to naive fusion when lineage is unrecorded. This is the gap the paper fills.
4. Value-of-provenance-information is promising on width but not yet on decisions — keep as a
   secondary result, re-test with a learned prior.

**Verdict: GO, with the reframed claim** (certificate + learned-prior point estimate + real lineage
ground truth from 5G signalling). Not GO for "duplication-invariant fusion" (published by PACT) or
"interval-based abstention" (refuted here).
