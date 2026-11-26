# Restricted-Bayes frontier between a reference mechanism and the worst case — protocol (written 2026-11-16T14:55:58Z, before any quantity below was computed)

**Status.** Analytical, post hoc: declared after the testbed and Flight results were known. No hypothesis is tested. The
construction is the restricted Bayes solution of Hodges and Lehmann (1952) applied to the copy-cap class; no novelty is claimed
for it. All earlier protocols, code and results are unchanged.

**Question.** A count rule trained on a stable copying mechanism can beat the minimax rule (Flight). How much worst-case
protection can be bought for how much loss under the reference mechanism?

**Definition.** Reference mechanism Q_ref in Delta_D; class Q_eps = {(1-eps) Q_ref + eps Q : Q in Delta_D} (label-aware Q).
Risk is linear in the mechanism, so sup over Q_eps of the Brier risk of a rule x is (1-eps) R_ref(x) + eps W_D(x), where
R_ref(x) = E_ref (x(A)-y)^2 needs only the reference law of (y, A), and W_D(x) is the worst case of Theorem 2, which needs the
law of (y, U). The restricted-Bayes rule x_eps minimises this convex quadratic objective. eps = 0 and eps = 1 are computed with
weights 1e-6 and 1 - 1e-6 so that ties are broken by the other criterion.

**Fixed before computing.**
* Grid: eps in {0, 0.05, 0.1, 0.2, 0.5, 1}. Solver: cvxpy/Clarabel, default tolerances, OMP_NUM_THREADS=1.
* Testbed: the 108 fold laws of results/finite_cap/fold_laws.pkl; reference mechanism = no duplication (A = U); D in {1, 4, 16};
  rule fitted on the training law, evaluated on the held-out law: reference risk and exact worst case.
* Flight: airline sites removed (mode B of docs/flight-addendum-R2.md), tau = 0.95, both splits, clusters and cap D_hat from
  training items only; reference law = empirical training law of (y, A); worst-case term from the training law of (y, U_cluster).
  Held-out: Brier on the observed counts (natural risk) and model worst case over Delta_{D_hat} for the held-out law of (y, U_cluster).
  The clusters are a lineage proxy, so the worst-case term is a model quantity, not an observed one.
* Reported per eps: mean reference/natural risk and mean worst-case risk (testbed: over 108 dependent folds, by D; Flight: pooled
  held-out items, by split, and by held-out group). No intervals across the three airlines.
* Descriptive reading fixed in advance: a level eps is called "cheap protection" if, against eps = 0, the natural (reference) risk
  rises by at most 0.010 Brier and at least half of the worst-case gap between eps = 0 and eps = 1 is closed. We report whether
  such a level exists in the grid, whichever way it turns out. Nothing is tuned on held-out data.
