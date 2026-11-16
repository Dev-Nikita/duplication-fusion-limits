# G3 pilot protocol — declared 2026-10-17, before the first run
Code: experiments/g3/run_g3.py; seed 20261019. Generator and scorer as G2 (hide 0.5 and 0.75);
starting space = G2 paef_cal sets at δ = 0.05 (calibration n = 1000); unit query costs; τ = 0.5.
Episodes with an undecided envelope at the start are the population (others need no query).
Repetitions: 30 per hide level (400 test episodes each).
Policies: random; max_uncertainty (entropy of π-marginal); expected_interval_reduction;
decision_robustness (myopic P(decisive after query), SDP-style); ec2 (equivalence classes =
{lo>τ}, {hi<τ}, {ambiguous}); dp_optimal (exact, episodes with ≤ 5 unresolved).
Metrics: queries to a robust decision (all unresolved queried = cap); share decided; error of the final
decision vs label and vs oracle-posterior decision; ratio to dp_optimal.
Hypotheses (fixed):
* Q1: decision_robustness and ec2 need fewer queries than random and max_uncertainty (paired bootstrap
  95% CI of the difference excludes 0).
* Q2: report the ratio of each policy to dp_optimal; no threshold fixed.
* Q3: share of undecided episodes in the single-unresolved-per-cluster regime (G3.1 applies).
* Gate: G3 has a *methods* contribution only if some policy we propose beats both ec2 and
  decision_robustness; none is proposed now, so a positive methods claim is excluded in advance.
