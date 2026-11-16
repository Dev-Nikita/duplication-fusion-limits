# Flight study — addendum R2: gold-source overlap and same-channel baselines (written 2026-11-16T07:19:28Z, before any of the quantities below was computed)

**Status.** Post-discovery sensitivity analysis, not a confirmatory test. The frozen protocol (docs/flight-protocol.md), its
code and its results are unchanged and remain the primary record.

**Why.** Li et al. (VLDB 2012, Sec. 2.2) built the gold standard of the Flight data from the websites of three airlines
(AA, UA, Continental). The same websites are among the sources (`aa`, `ua`, `CO`). The frozen study did not exclude
them, so a rule could read a report that is the gold value itself. A second concern: the rule called "naive" is trained on
cluster counts U and applied to raw counts A; it is a channel-mismatched rule, and same-channel simple baselines were missing.

**Fixed before computing.**
1. Overlap audit: per gold item, whether the item's own airline site reports it; agreement (within 1 min) of every source's
   actual departure with the gold value, by airline; agreement of sources' scheduled departure with the gold schedule.
2. Exclusion modes, each rerun through the unchanged pipeline (clusters, caps, features, laws and models rebuilt on training
   items only): **C** original; **A** the item's own airline site removed from that item (and from cluster building for that
   airline's flights); **B** the three airline sites removed everywhere. No item is dropped. Same thresholds, tau grid, seeds,
   splits and bootstrap as the frozen protocol. H-F1/H-F2/H-F3 are re-read under A and B with the original criteria.
3. Additional same-channel rules (primary tau, all modes): `table_A` (empirical P(y | A) with the frozen table rule, trained and
   applied on A); `frac` (GBDT on A/(A+N), -1 if no source reports); `coverage` (GBDT on A+N); `neg_only` (GBDT on N);
   `majority` (hard vote 1{A > N}); `frac_table` (Laplace-smoothed table over ten equal-width bins of A/(A+N));
   `own_site` (mode C only: GBDT on the own airline site's signal alone: fires / negative / silent).
4. Per-airline and per-date-block Brier scores of count, unique, count_neg and frac. With three airlines no interval across
   airlines is computed; the day-cluster bootstrap stays descriptive.
5. Reading rule fixed in advance: if a same-channel rule on A and N without lineage (frac, frac_table or majority) reaches the
   Brier score of count_neg within 0.005, the gain of negative reports is attributed to plain normalisation by coverage and the
   text says so. If results under A or B differ from C by more than 0.005 in a primary contrast, the manuscript reports A or B
   as the main table and C as the leakage-prone variant.
