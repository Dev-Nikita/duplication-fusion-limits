# Estimation-error analysis of finite-cap fusion rules — protocol (frozen 2026-10-26, before running)

Purpose: §6.4 showed that the plug-in finite-cap minimax rule beats the naive count rule in 104/108 held-out
folds at D=16 but beats the presence rule in only 38/108. Copy uncertainty and estimation error of P(y,U) are
different sources of risk. This analysis asks, on existing data only, which simple fitting rule controls both.
It is **analytical and post hoc on previously analysed data**; it is not confirmatory. Nothing is tuned after
running: rules, constants (alpha = 0.5; eps rule; D grid) and summaries are fixed here.

Folds: the 108 outer folds of `results/finite_cap/fold_laws.pkl` (R10, R6, R11-control, R11-loaded; three primary
tasks; leave-one-block-out; one report type, M = 20). D in {1, 2, 4, 8, 16}.

Rules (fitted on training blocks only):
1. plugin — finite-cap minimax rule of the pooled training law;
2. presence — Bayes rule of presence on the pooled training law;
3. smoothed — finite-cap minimax rule of the add-0.5 smoothed law (42 cells);
4. tvrobust — min_x L_D(x; p̂) + eps · max_{y,u} c_yu(x), eps = mean over training blocks of TV(block law, pooled
   training law) (an upper bound on the worst case over a TV ball; eps is a heuristic block-variability scale,
   **not** a coverage-guaranteed radius);
5. select — plugin or presence, chosen by the lower mean exact worst-case risk over inner leave-one-training-block-
   out folds.

Outcome: exact worst-case Brier risk over Delta_D on the held-out block law (W).

Summaries: mean W per dataset × task × D; per-fold wins against presence and against plugin; regret of each rule
relative to the per-fold better of {plugin, presence} (an oracle choice not available in practice).

Expectations written before running (reported whether they hold or not):
* E1: select has mean regret ≤ 0.01 relative to the per-fold oracle choice at every D (pooled over 108 folds).
* E2: tvrobust has lower W than plugin in more than half of the folds at D = 16.
* E3: smoothed differs from plugin by less than 0.005 in mean W at every D ≤ 4.

Code and inputs (sha256):
```
1b40ab4361de4d99cce60011108b89d84bc9bd11a860304d87a68fd538396f1d  experiments/est/estimation_error.py
3517932d8767a38bb0de45bd160c4c450a95a70fe3bdb6953303418b7552a1d9  tools/finite_cap_cvx.py
f6b9fbd5c9bb6837232407651e1fbecbcfef80ac02407817e1030faabfeee98b  tools/finite_cap_solvers.py
c6f5e9764b95ac5a24073fbf42acaa12ca716db8da4650c6f0d65ea423990964  tools/fold_laws.py
d76503d2b724c9574a537ca6f56d93d3c7498e883f568771cd03d83b91da06d0  results/finite_cap/fold_laws.pkl
```

## Addendum B (declared 2026-11-15T18:00:35Z, before running; post hoc on previously analysed folds)
Purpose: stronger lineage-free baselines for the finite-cap comparison (the naive count rule is a weak reference).
Rules, each built from the training law η(u) = P̂(y=1 | U=u) and the known cap D, applied to the raw count a:
1. naive: η(min(a, M));
2. floor: η(⌈a/(D+1)⌉) (assumes maximal copying);
3. mid: η(round(2a/(D+2))) clipped to [⌈a/(D+1)⌉, min(a, M)] (assumes average multiplicity (D+2)/2);
4. interval-mean: training-mass-weighted mean of η(u) over the consistent set u ∈ [⌈a/(D+1)⌉, min(a, M)];
5. interval-midrange: (min η + max η)/2 over the consistent set;
6. presence; 7. plug-in minimax (interior point); 8. select (as in the main protocol).
Outcome: exact worst-case Brier over Δ_D on the held-out block law, D ∈ {1, 4, 16}, and the no-copy risk; means over
the 108 dependent folds and fold counts (descriptive). No tuning; no further rules added after seeing results.
Code: experiments/est/baselines.py.
