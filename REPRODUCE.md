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
| `make audit` | checks the manifest checksums and the repository hygiene script | seconds |

## What each step produces
| Step | Command | Output | Used for |
|---|---|---|---|
| 1 | `python3 tools/test_finite_cap_independent.py` | 94 checks | Theorem 2 (two independent solvers) |
| 2 | `python3 tools/saturation_closed_form.py` | 4,302 checks | Proposition 4 (closed form against the LP) |
| 3 | `python3 -W ignore tools/quantitative_saturation.py` | 2,993 checks | Proposition 5 (bounds, attainment, Lipschitz) |
| 4 | `python3 -W ignore tools/extended_models.py` | 579 checks | Propositions 6–7 (mechanism information, several types) |
| 5 | `python3 -m pytest -q tests` | 409 tests | code base and adversarial edge cases (`tests/test_theory_adversarial.py`) |
| 6 | `tools/apply_finite_cap.py` (two calls, see `tools/clean_room_run.sh`) | `results/finite_cap/part{1,2}/` | mirror-ascent brackets (independent check of R*_D) |
| 7 | `tools/presence_rule_oos.py`, `tools/fold_laws.py`, `tools/plugin_cvx_oos.py`, `tools/li_vs_ld.py`, `tools/li_certify.py` | `results/finite_cap/*.csv`, `fold_laws.pkl` | Table 4 inputs; count-only game and its certificate |
| 8 | `python3 tools/saturation_closed_form.py --folds results/finite_cap/fold_laws.pkl results/finite_cap/saturation_closed_form.csv` | D_sat per fold | Table 4 |
| 9 | `python3 tools/summarise_finite_cap.py`; `python3 tools/make_table4.py` | `summary.csv`, `table4_rows.tex`, `paper/figures/fig_copycap.pdf` | Table 4, Figure 1, fold counts of Section 7.3 |
| 10 | `python3 -W ignore tools/block_sensitivity.py` | `block_*.csv` | block-level and leave-one-physical-block-out results (Section 7.3) |
| 11 | `python3 -W ignore tools/saturation_stability.py 500`; `python3 tools/make_stability_table.py` | `saturation_stability_*.csv`, `stability_rows.tex` | Table 5 |
| 12 | `python3 -W ignore experiments/est/estimation_error.py results/finite_cap/fold_laws.pkl results/est/estimation_error.csv` | estimation-error study | Section 7.4, Table 6 (protocol `docs/est-protocol.md`) |
| 13 | `python3 -W ignore experiments/est/baselines.py results/finite_cap/fold_laws.pkl results/est/baselines.csv`; `python3 tools/make_baseline_table.py` | `baselines.csv`, `baseline_rows.tex` | Table 6 (protocol addendum B) |
| 14 | `python3 -W ignore tools/exchangeability_check.py`; `python3 -W ignore tools/multitype_data.py` | `exchangeability.csv`, `multitype.csv` | Section 7.5 |
| 15 | `python3 experiments/paper/figure1_values.py` | `results/editorial/fig1_channel_gaps.csv` | Figure S1, Table S3 |
| 16 | `python3 -I -W ignore experiments/flight/flight_study.py <clean_flight dir> <flight_truth dir> results/flight`; `python3 -I experiments/flight/flight_evaluate.py results/flight` | `results/flight/*` | frozen Flight run, Tables S7–S8 (protocol `docs/flight-protocol.md`) |
| 17 | `python3 -I -W ignore experiments/flight/flight_audit.py <clean_flight dir> <flight_truth dir> results/flight/audit_R2`; `python3 -I -W ignore experiments/flight/flight_audit_descriptors.py <clean_flight dir> <flight_truth dir> results/flight/audit_R2` | `results/flight/audit_R2/*` | gold-source audit and same-channel baselines: Table 7, Tables S5–S6 (addendum `docs/flight-addendum-R2.md`, written before computing) |
| 18 | `python3 -W ignore tools/cross_platform_check.py 0 108` | one line: largest deviation from the stored reference | cross-platform check of the 756 finite-cap values |

Table 3 (R11-X) and Tables S1–S4 come from the campaign analyses: `experiments/r11x/replicate.py` and
`experiments/r11x/evaluate_r11x.py` on `results/r10/units.csv` and `results/r6/units.csv` (outputs in
`results/r11x_rep/`), `experiments/r11x/robust_seeds.py` (`results/r11x_seeds10/`), and `experiments/sim/`
(`results/sim/`). Their protocols are in `docs/` with the sha256 of the code they froze.

## External data
* Flight (Li et al., VLDB 2012): `clean_flight.zip` and `flight_truth.zip` from
  http://lunadong.com/fusionDataSets.htm. sha256 values are in `docs/flight-protocol.md`'s data note
  (`data_external/lunadong_fusion/SHA256SUMS`). Unzip both and pass the two directories to steps 16 and 17. The site states no
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
