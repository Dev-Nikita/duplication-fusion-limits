# Supplementary sensitivity analysis — results (docs/sim-protocol.md; S1, S2 amendments)
192 cells × 5 replications, 4,000 training and 4,000 test windows each. Outcome: Brier loss relative to the
distinct-unit (exact-lineage) fuser, worst case over the mirrored LD pair. Cell class (δ = 0.005, t-interval
over 5 replications): works (upper < δ), fails (lower > δ), undetermined. Files: results/sim/.

## Class counts
| Channel | works | undetermined | fails |
|---|---|---|---|
| A (firing messages) | 0 | 1 | 191 |
| F (firing messages + negative reports) | 35 | 29 | 128 |
| A+U (firing messages + oracle distinct units) | 10 | 71 | 111 |

## Pre-written expectations
| | Expectation | Outcome |
|---|---|---|
| E1 | A's loss grows with κ, ≈ 0 at κ = 0 | **Partly.** It grows with κ (mean 0.21 → 0.29 → 0.37). It is **not** ≈ 0 at κ = 0 (0.09–0.39 with known M): storms of 16 copies lie outside the augmentation support (D ≤ 8), so multiplicity shift alone is costly |
| E2 | F ≈ 0 at q = 0, known M, sil = 0; grows with q | **Partly.** F is identical to A+U there, which confirms the sufficiency (U = M − N). But it is **not** ≈ 0: 0.001–0.071 at κ = 0.6. It grows steeply with q (κ = 0.6, ρ = 0.3: 0.07 → 0.29 → 0.44 → 0.46) |
| E3 | Unknown population hurts F most at small ρ | **Partly.** It hurts F everywhere (κ = 0.6, q = 0: 0.13–0.26 vs 0.001–0.07), and most at ρ = 0.3–0.6, not at the smallest ρ |
| E4 | Silencing changes F's loss, direction open | **Holds.** With sil = 0.5 and low q, F *beats* the distinct-unit fuser (down to −0.08). Missing reports from silenced units carry state information that lineage does not see (Regime III) |
| E5 | A+U ≈ 0 everywhere | **Does not hold.** Mean +0.038, max +0.110. Adding the distinct-unit count does not neutralise the raw firing count |

## Exploratory follow-up (after the frozen run; experiments/sim/explore_nonly.py; κ = 0.6, sil = 0)
Channel N: the fuser gets negative reports (and M) **instead of** firing counts.

| | known M, q = 0 | known M, q = 0.3 | unknown M, q = 0 |
|---|---|---|---|
| N (denominator instead of counts), ρ = 0.1 / 0.3 / 0.6 / 0.9 | **0.000 / 0.000 / 0.000 / 0.000** | 0.035 / 0.090 / 0.056 / 0.013 | 0.026 / 0.080 / 0.060 / 0.013 |
| F (denominator added to counts), same ρ | 0.054 / 0.071 / 0.026 / 0.001 | 0.206 / 0.443 / 0.504 / 0.399 | 0.131 / 0.264 / 0.238 / 0.124 |

## Reading
1. **Sufficiency is not robustness.** A channel can carry what is needed to recover the multiplicity of
   independent evidence (F, A+U) and still give a fuser that fails under an unseen duplication mechanism.
   The reason: the fuser also uses the mechanism-sensitive firing count. Replacing the counts by an invariant
   statistic (distinct units, or negative reports with known M) removes the loss in Regime I. Under report
   loss or an unknown population it reduces the loss 3- to 10-fold relative to F.
2. **The same pattern on the testbed:** lineage-separated features (distinct units *plus* copy counts) left a
   residual (X2 not confirmed: +0.003/+0.005). In R8 HA, lineage-aware models trusted invisible duplicates.
   Copy *removal* is invariant; provenance supplied as extra features is not.
3. **Regime III appears as predicted in kind:** state-dependent silence makes the missing count informative,
   and lineage cannot see it.
4. **Difference from the testbed:** there, F was close to exact lineage (R10 +0.001), while in the simulation
   F is not. That GBDT apparently relied on the non-firing counts. The simulation shows this is not
   guaranteed, so the testbed's F result should not be generalised.
5. **Limits:** a single generator with one reading type, one learner configuration and synthetic duplication.
   It is a second kind of evidence, not an independent confirmation, and E1, E2 and E5 were partly or wholly
   not met.

## Critical check: is the residual a training-support artefact? (exploratory, experiments/sim/explore_support.py)
Known M, q = 0, sil = 0. Augmentation support narrow, D ≤ 8 (as frozen), or wide, D ≤ 32 (covers the test storm
of 16). Mean worst-case loss vs U over ρ ∈ {0.1, 0.3, 0.6, 0.9}:

| Channel | narrow, κ = 0 | narrow, κ = 0.6 | wide, κ = 0 | wide, κ = 0.6 |
|---|---|---|---|---|
| A (firing counts) | 0.10–0.40 | 0.24–0.63 | **0.04–0.27** | **0.10–0.43** |
| log(1 + A) | same as A | same as A | same as A | same as A |
| F (A + negatives) | 0.002–0.022 | 0.001–0.076 | **0.000–0.005** | **0.001–0.009** |
| N (negatives only) | 0.000 | 0.000 | 0.000 | 0.000 |

**Consequences:**
* Most of F's residual, the "sufficiency ≠ robustness" pattern, is an artefact of training support. Wider
  augmentation removes most of it. It must **not** be the headline.
* The firing-count channel is not robust to a change of duplication mechanism, *even when the mechanism is
  label-independent and inside the training support*. P(y | A) depends on the multiplicity distribution,
  because A confounds the number of independent positives with the copy volume. Label dependence adds to the
  loss but is not required for it. This is the robust core finding. It is elementary (non-identifiability of
  a mixture), not new theory.
