# Pilot protocol P1–P4 (synthetic) — declared before the first full run

Date: 2026-10-17. Code: `experiments/pilot/run_pilot.py`, seed 20261017, n = 3000 episodes per cell.
Purpose: decide whether the incomplete-provenance interval idea has enough substance to justify
the 5G testbed. This is a go/no-go gate, not a paper result.

## Generative model (pilot)
H → U_r (origin signal of root r) → Z_i. Exact copies replicate a parent observation; derived
observations are new noisy readings of U_r. A fraction `hide` of secondary observations has
candidate set {true origin, 1 decoy, FRESH}. `misspec` drops the true origin from the set.

## Methods
naive_bayes · declared_only (unresolved ⇒ independent) · pact_meet_declared (PACT-style meet,
declared partition only) · pact_meet_closure (meet over the candidate closure) ·
cautious_closure (max-rule analogue of the cautious rule) · **paef_bma** (uniform model average
over admissible lineages) · **paef_interval** [lower, upper] · oracle (true lineage; reference only).
The PACT and cautious rules are *binary log-LR adaptations* written for this pilot, not the
authors' implementations; this must be stated wherever they are reported.

## Hypotheses and kill criteria (fixed now)
* S0 (sanity): with misspec = 0 the oracle posterior lies inside [lower, upper] in 100% of episodes.
* H1 (duplication): with *known* copies, every lineage-aware method has DS_k = 0; with *unknown*
  copies the PAEF lower bound has DS_k = 0 for agreeing evidence while naive/declared_only grow with k.
* H2 (selective): at hide ≥ 0.5 the interval rule's error on the cases it decides is lower than the
  error of every non-oracle point method at the same coverage.  **K1: if not, the selective claim is dropped.**
* H3 (point): paef_bma has lower Brier than declared_only and both PACT variants at hide ≥ 0.5.
  **K2: if not, the point estimate is not claimed.**
* H4 (informativeness): at hide = 0.5 the interval decides ≥ 50% of episodes.
  **K3: if below, the interval is too vacuous to be useful as stated.**
* H5 (value of provenance information): resolving the link chosen by expected width reduction
  decides more straddling episodes than resolving a random unresolved link (bootstrap CIs disjoint).
* P3 (misspecification) is descriptive: report how coverage and risk degrade; no claim fixed.

Nothing in this file is edited after the run; deviations go to `docs/amendments.md`.
