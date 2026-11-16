# Minimax information limits of evidence fusion under unknown report duplication — code, protocols, results

Code, frozen protocols, testbed definition and derived results for the manuscript
*"Minimax Information Limits of Evidence Fusion under Unknown Report Duplication"* (Nikita Tarasov, Lviv Polytechnic
National University). The manuscript itself is not part of this repository (see `PRIVATE_FILES.md`).

## Layout
| Path | Contents |
|---|---|
| `testbed/` | Docker definition of Open5GS v2.7.6 + UERANSIM v3.2.7, fault injection (`scripts/campaign.py`), `run.sh`, `RUNBOOK.md` |
| `src/paef/` | fusion methods, duplication mechanisms, per-UE layer, baselines |
| `experiments/` | per-campaign feature extraction and analysis (R4–R12, R11-X, simulation, RCAEval, estimation-error study and baselines in `est/`, Flight benchmark in `flight/`); `experiments/paper/` figures |
| `tools/` | finite-cap theory: solvers (`finite_cap_solvers.py`, `finite_cap_cvx.py`, `extended_models.py`), verification (`test_finite_cap_independent.py`, `saturation_closed_form.py`, `quantitative_saturation.py`, `li_certify.py`), application to data, tables, figures and the release manifest (`make_manifest.py`); repository hygiene (`git_safety_check.sh`, `make_public_release.sh`) |
| `docs/` | frozen protocols (`*-protocol.md`, with sha256 of the code they froze), `amendments.md`, result reports |
| `results/` | derived per-UE units, per-cell results and every intermediate file behind the reported numbers |
| `tests/` | unit and property tests (`python -m pytest -q tests`) |

## Reproduce
`REPRODUCE.md` lists every step; `MANIFEST.csv` maps each table and figure of the article to its inputs, command,
output files and sha256; `make reproduce-math`, `make reproduce-tables`, `make reproduce-figures`,
`make reproduce-external`, `make cross-platform`, `make clean-room DIR=<empty dir>` and `make audit` run them
(`pip install -r requirements_finite_cap.txt` first). Testbed (macOS/Linux with Docker, about 25 min per block):
`cd testbed && ./run.sh build`, then the campaign targets in `RUNBOOK.md`.

## Status of analyses
Campaign protocols were frozen (sha256 of code) before data or analysis existed; amendments are in `docs/amendments.md`.
The Flight study was frozen before any outcome was computed (`docs/flight-protocol.md`); its primary hypothesis was not
confirmed. After the frozen run we found that three airline websites are both sources and the providers of the gold
standard; `docs/flight-addendum-R2.md` (written before computing) defines the rerun without them and the added
same-channel baselines. That rerun is a sensitivity analysis, not a confirmatory test; both sets of results are kept
(`results/flight/`, `results/flight/audit_R2/`). The finite-cap application, the baselines and the mechanism-information and second-type analyses are
analytical and were declared after the testbed data had been seen. Exploratory analyses are labelled as such.
The archive `paef-provenance-release.zip` of an earlier project stage is superseded by this layout.

## Licence and citation
To be added with the public release (DOI on acceptance).
