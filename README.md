# Evidence fusion under unknown report duplication — code, protocols and derived results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24561927.svg)](https://doi.org/10.5281/zenodo.24561927)

Code, testbed definitions, frozen experiment protocols and derived results for a study of fusion of counted reports
when reports may be duplicated by an unknown mechanism (Nikita Tarasov and Roman Zinko, Lviv Polytechnic National
University).

## Layout
| Path | Contents |
|---|---|
| `testbed/` | Docker definition of an Open5GS v2.7.6 + UERANSIM v3.2.7 5G core testbed, fault injection, `run.sh`, `RUNBOOK.md` |
| `src/paef/` | fusion methods, duplication mechanisms, per-UE layer, baselines |
| `experiments/` | per-campaign feature extraction and analysis; Flight benchmark (`flight/`); broker replay through NATS JetStream (`redelivery/`); two-type testbed with fault proxies and broker replay (`multitype/`, Go) |
| `tools/` | finite-cap solvers and verification, whole-report copying (`whole_report_saturation/`, exact rational arithmetic), K=2 evaluation (`whole_report_eval/`), data-integrity gate, tables, figures, release manifest |
| `docs/` | frozen protocols (`*-protocol.md`, with sha256 of the code they froze), `amendments.md`, result reports |
| `results/` | derived per-unit readings and every intermediate result behind the reported numbers |
| `tests/` | unit and property tests (`python -m pytest -q tests`) |

## Reproduce
`pip install -r requirements_finite_cap.txt`, then see `REPRODUCE.md`. `MANIFEST.csv` maps each table and figure to its
inputs, command, outputs and sha256. Main targets: `make reproduce-math`, `make reproduce-tables`,
`make reproduce-figures`, `make reproduce-external`, `make integrity`, `make whole-report-quick`,
`make whole-report-eval`, `make audit`. Testbed (Docker): `cd testbed && ./run.sh build`, then `RUNBOOK.md`.
Two-type testbed: `experiments/multitype/README.md`.

## Status of analyses
Campaign protocols were frozen (sha256 of code) before data or analysis existed; amendments are in
`docs/amendments.md`. Hypotheses that were not confirmed are reported as such. Three invalid testbed blocks (R6
blocks 7–9) were found in the input of the broker replay; the corrected analysis is in `results/redelivery_corrected/`,
the original one is kept and marked withdrawn (`results/redelivery/WITHDRAWN.txt`), and `make integrity` checks that
no other result uses those blocks. Exploratory analyses are labelled as such.

## Data
Included: per-unit testbed readings and all derived results. Not included: packet captures and raw logs (about 35 GB,
available on request) and external datasets (Flight, Li et al. 2012; RCAEval, Zenodo record 14504481), which
`REPRODUCE.md` explains how to download.

## Licence and citation
Code: MIT (`LICENSE`). Derived data in `results/`: CC BY 4.0. Third-party data and software: `DATA_LICENSES.md`.
Archived on Zenodo: https://doi.org/10.5281/zenodo.24561927. Citation metadata: `CITATION.cff`.
