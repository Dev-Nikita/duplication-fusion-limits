# Supplementary sensitivity analysis on a generative model — protocol (frozen 2026-10-24, before running)
Purpose (reviewer request, 2026-10-24): to map, outside the testbed, when a lineage-free learner can recover the
multiplicity of independent evidence from the observation channel, and when it cannot. This analysis is
**supplementary**: it changes no R4–R11-X result or conclusion, and it is not a confirmation of them. Nothing is
tuned toward a result: the grid, the learner and its hyperparameters are fixed here.

Model and channels: see the docstring of experiments/sim/sensitivity.py.
Axes, full factorial, 192 cells × 5 replications:
* fraction of affected units ρ ∈ {0.1, 0.3, 0.6, 0.9};
* label dependence of duplication κ ∈ {0, 0.3, 0.6};
* loss of negative reports q ∈ {0, 0.1, 0.3, 0.6};
* population known (M = 20) or unknown (M ~ U{10..20});
* silencing of affected reporters sil ∈ {0, 0.5}.

Outcome: Brier loss relative to the exact-lineage (distinct-unit) model, worst case over the mirrored LD pair,
for the channels A, F and A+U.

Expectations written before running (descriptive; each is reported whether it holds or not):
* E1: A's loss grows with κ and is ≈ 0 at κ = 0.
* E2: F's loss is ≈ 0 at q = 0, population known, sil = 0, and grows with q.
* E3: an unknown population hurts F most when ρ is small (firing fractions not near 0 or 1).
* E4: silencing (negative reports missing *because of* the fault) changes F's loss, in a direction not
  predicted.
* E5: A+U ≈ 0 everywhere.

Code (sha256):
```
dc7e1218ade532998aebdfefa59802861c7dae57fbf6ad00ade12c1bbaf3a08f  experiments/sim/sensitivity.py
```

Amendment S1 (2026-10-24, before any result was inspected). The first run was stopped after 2 of 192 cells
because the per-window loop was too slow (~22 h). No output was read. sample() was vectorised over windows:
same model, same grid, same learner. Random streams differ from the stopped run. Checked: with q = 0, sil = 0
and known M, U + N = M in every window; A ≥ U. New sha256:
```
281bee2f180fb88ddb025a4489ba329712a21e8612bd5545b2cc2c2251d0dcb4  experiments/sim/sensitivity.py
```

Amendment S2 (2026-10-24, no result inspected). The run is executed through experiments/sim/run_resumable.py,
which imports sensitivity.py (hash above, unchanged) and repeats its cell loop with identical seeds. It appends
results per cell, so a stopped run resumes. Execution changes: OMP_NUM_THREADS=1, because two jobs share two
cores. HistGradientBoosting results do not depend on the thread count at a fixed random_state.
```
8d0a20f51da36dc608fd21e7b9c5498ba1ab1287c010b47f3b2d118a7ce5ab7f  experiments/sim/run_resumable.py
```
