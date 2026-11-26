# Reproducing the results

Requirements: Python 3.10+ and `pip install -r requirements_finite_cap.txt` (numpy, pandas, scipy, scikit-learn,
matplotlib, cvxpy with the Clarabel solver, pytest). Run everything from the project root with `OMP_NUM_THREADS=1`.
`MANIFEST.csv` maps every table and figure of the article to its inputs, command, output files and sha256.

## One-command targets (`make <target>`)
| Target | What it does | Time |
|---|---|---|
| `make reproduce-math` | all numerical checks of the theory (Theorems 1–2, Propositions 4–7) and the unit tests | about 3 min |
| `make reproduce-tables` | regenerates the bodies of Tables 4–6 from stored intermediate results | under 1 min |
| `make reproduce-figures` | Figure 1 (copy cap) and the values of Figure S1 | under 1 min |
| `make reproduce-external` | Flight benchmark: frozen run, then the gold-source audit (needs the public Flight archives, see below) | about 6 min |
| `make cross-platform` | recomputes the 756 finite-cap values and compares them with the stored reference and the mirror-ascent brackets | under 1 min |
| `make clean-room DIR=<empty dir>` | recomputes the whole finite-cap analysis from the per-UE readings and compares every result file byte by byte | 1–2.5 h |
| `make audit` | regenerates the three table bodies (`*.tex`, not stored in the archive), then checks the manifest checksums and, in a Git checkout, the hygiene script | under 1 min |

## What each step produces
| Step | Command | Output | Used for |
|---|---|---|---|
| 1 | `python3 tools/test_finite_cap_independent.py` | 94 checks | Theorem 2 (two independent solvers) |
| 2 | `python3 tools/saturation_closed_form.py` | 4,302 checks | Proposition 4 (closed form against the LP) |
| 3 | `python3 -W ignore tools/quantitative_saturation.py` | 2,993 checks | Proposition 5 (bounds, attainment, Lipschitz) |
| 4 | `python3 -W ignore tools/extended_models.py` | 579 checks | Propositions 6–7 (mechanism information, several types) |
| 5 | `python3 -m pytest -q tests` | all tests pass (410+6; the count depends on optional dependencies) | code base and adversarial edge cases (`tests/test_theory_adversarial.py`) |
| 6 | `tools/apply_finite_cap.py` (two calls, see `tools/clean_room_run.sh`) | `results/finite_cap/part{1,2}/` | mirror-ascent brackets (independent check of R*_D) |
| 7 | `tools/presence_rule_oos.py`, `tools/fold_laws.py`, `tools/plugin_cvx_oos.py`, `tools/li_vs_ld.py`, `tools/li_certify.py` | `results/finite_cap/*.csv`, `fold_laws.pkl` | Table 4 inputs; count-only game and its certificate |
| 8 | `python3 tools/saturation_closed_form.py --folds results/finite_cap/fold_laws.pkl results/finite_cap/saturation_closed_form.csv` | D_sat per fold | Table 4 |
| 9 | `python3 tools/summarise_finite_cap.py`; `python3 tools/make_table4.py` | `summary.csv`, `table4_rows.tex` | Table 4, fold counts of Section 7.3 |
| 10 | `python3 -W ignore tools/block_sensitivity.py` | `block_*.csv` | block-level and leave-one-physical-block-out results (Section 7.3) |
| 11 | `python3 -W ignore tools/saturation_stability.py 500`; `python3 tools/make_stability_table.py` | `saturation_stability_*.csv`, `stability_rows.tex` | Table 5 |
| 12 | `python3 -W ignore experiments/est/estimation_error.py results/finite_cap/fold_laws.pkl results/est/estimation_error.csv` | estimation-error study | Section 7.4, Table 6 (protocol `docs/est-protocol.md`) |
| 13 | `python3 -W ignore experiments/est/baselines.py results/finite_cap/fold_laws.pkl results/est/baselines.csv`; `python3 tools/make_baseline_table.py` | `baselines.csv`, `baseline_rows.tex` | Table 6 (protocol addendum B) |
| 14 | `python3 -W ignore tools/exchangeability_check.py`; `python3 -W ignore tools/multitype_data.py` | `exchangeability.csv`, `multitype.csv` | Section 7.5 |
| 15 | `python3 experiments/paper/figure1_values.py` | `results/editorial/fig1_channel_gaps.csv` | Figure S1, Table S3 |
| 16 | `python3 -I -W ignore experiments/flight/flight_study.py <clean_flight dir> <flight_truth dir> results/flight`; `python3 -I experiments/flight/flight_evaluate.py results/flight` | `results/flight/*` | frozen Flight run, Tables S7–S8 (protocol `docs/flight-protocol.md`) |
| 17 | `python3 -I -W ignore experiments/flight/flight_audit.py <clean_flight dir> <flight_truth dir> results/flight/audit_R2`; `python3 -I -W ignore experiments/flight/flight_audit_descriptors.py <clean_flight dir> <flight_truth dir> results/flight/audit_R2` | `results/flight/audit_R2/*` | gold-source audit and same-channel baselines: Table 7, Tables S5–S6 (addendum `docs/flight-addendum-R2.md`, written before computing) |
| 18 | `python3 -W ignore tools/cross_platform_check.py 0 108` | one line: largest deviation from the stored reference | cross-platform check of the 756 finite-cap values |
| 19 | `python3 -W ignore tools/frontier.py testbed`; `python3 -W ignore tools/frontier.py flight <clean_flight dir> <flight_truth dir>` | `results/frontier/*` | restricted-Bayes frontier: Remark 3, Table 8 (protocol `docs/frontier-protocol.md`, written before computing; analytical) |
| 20 | `python3 -W ignore tools/whole_report_marginal.py`; `python3 -m pytest -q tests/r4`; `python3 -W ignore tests/r4/test_exact_brackets.py` | printed summaries; `results/r4/exact_brackets.txt` | marginal counts under whole-report copying (Proposition 3); exact rational checks of Propositions 4 and 5 |
| 21 | `cd experiments/redelivery && python3 -W ignore evaluate.py confirm/out input/truth.csv ../../tools ../../results/redelivery` | `results/redelivery/*` | broker redelivery (Section 8.4): audit of the recorded trace and evaluation. The trace itself comes from one recorded run (`run.sh`, Docker and NATS JetStream); rerunning the broker gives a new trace, not the recorded one |
| 22 | `cd experiments/redelivery && python3 -W ignore robust_risk.py confirm/out input/truth.csv ../../tools ../../results/redelivery` | `results/redelivery/robust_risk.csv`, `per_block.csv` | post hoc: exact model worst case of every fitted rule (label-aware and count-only), caps 0–3, results by block (frozen analysis; superseded by step 25) |
| 23 | `make whole-report-quick` (about 1 min) | `results/whole_report/out/*` | whole-report copying: two-type thresholds, Proposition 4 at K=1, adjacency, signature cells, independent verification of the 1,410 KKT certificates, edge-case tests (Supplementary S8) |
| 24 | `make whole-report-full` (up to about an hour) | `results/whole_report/out/*` | Theorem 3 on 315 laws, independent LP/CLARABEL checks, certificates, Proposition 8 gap checks, three-type rule |
| 25 | `make redelivery-corrected` | `results/redelivery_corrected/{A,B}/*` | broker replay without the invalid R6 blocks 7–9 (variant B, used in the article: threshold recomputed on valid blocks; A: threshold as recorded); Tables 8–9, S11–S14 |
| 26 | `make integrity` | `results/integrity/*` | data-integrity gate: block validity registry; invalid R6 blocks 7–9 absent from every result outside the raw readings and the withdrawn replay; split overlap; Flight gold sites removed; stated episode counts |
| 27 | `python3 -W ignore tools/label_blind_saturation.py folds results/label_blind/folds.csv`; `python3 -W ignore tools/label_blind_saturation.py random 0 100` (and 100 200, 200 300) | `results/label_blind/*` | Corollary 2 (count-only saturation): testbed folds and 300 random laws |
| 28 | `cd results/whole_report && python3 ../../tools/whole_report_saturation/tolerance_caps.py` | `results/whole_report/out/tolerance_caps.csv` | exact and certified tolerance caps (Table S18) |

Table 3 (R11-X) and Tables S1–S4 come from the campaign analyses: `experiments/r11x/replicate.py` and
`experiments/r11x/evaluate_r11x.py` on `results/r10/units.csv` and `results/r6/units.csv` (outputs in
`results/r11x_rep/`), `experiments/r11x/robust_seeds.py` (`results/r11x_seeds10/`), and `experiments/sim/`
(`results/sim/`). Their protocols are in `docs/` with the sha256 of the code they froze.

* `make whole-report-eval` — exploratory K=2 dry run on the existing testbed: per-UE two-type whole reports,
  108 leave-one-block-out folds; writes `results/whole_report_eval/dryrun_folds.csv` and `dryrun_channels.csv`
  (post hoc, not a hypothesis test). The K=2 experiment code is in `experiments/multitype/` (not yet run on a server).

* `make reproduce-figures` (or `python3 -W ignore tools/paper_figures.py`) — every figure of the article and the
  supplementary material drawn with matplotlib from files in `results/` into `results/figures/` (the worked example of
  the saturation figure is computed exactly by the script; its values are written to `fig_saturation_values.json`).
  The two schematic figures (observation model, whole-report geometry) are drawn in the manuscript source.

## External data
* Flight (Li et al., VLDB 2012): `clean_flight.zip` and `flight_truth.zip` from
  http://lunadong.com/fusionDataSets.htm. sha256 values are in `docs/flight-protocol.md`'s data note
  (`data_external/lunadong_fusion/SHA256SUMS`). Unzip both and pass the two directories to steps 16, 17 and 19. The site states no
  licence; the archives are not redistributed here.
* RCAEval (Zenodo record 14504481, CC-BY-4.0): RE3 and RE2; scripts in `experiments/ext/`.

## What cannot be regenerated from this archive
* The testbed campaigns themselves (2,100 episodes): they need the Docker testbed in `testbed/` and about 25 minutes
  per block. Packet captures and raw logs are not in the archive (size); the per-UE readings derived from them are
  (`results/*/units*.csv`). A clean-room run starts from those readings.
* Byte-identical results are expected on the same library versions (`requirements_lock_2026-11-16.txt` lists the
  versions of the reference run; `make versions` prints yours). Across platforms the claim is agreement to solver
  tolerance (1e-4), not identical floating-point output. Checked so far: Linux x86_64 (Python 3.13) and Linux aarch64
  (Python 3.10, older numpy/scipy/cvxpy, same Clarabel 0.11.1); see `results/finite_cap/cross_platform_2026-11-16.txt`.
  Native macOS and Windows have not been run.

## Which solver produced which number
All reported finite-cap values and rules are interior-point solutions (`tools/finite_cap_cvx.py`). Mirror ascent
(`tools/finite_cap_solvers.py`) is an independent check: in all 756 fold–cap cells the interior-point value lies inside
the mirror-ascent bracket. The count-only game is certified by a dual lower bound (`tools/li_certify.py`).
