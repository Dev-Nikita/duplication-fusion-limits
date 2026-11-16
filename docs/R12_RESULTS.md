# R12 results — RCAEval RE2 (protocol docs/r12-protocol.md, frozen 2026-10-24; amendment A1)

## Data and integrity
* RE2.zip md5 6f84a83c4fe1c0d7eb08e131775687bf, identical to the Zenodo-published checksum.
* All 540 logs.csv and inject_time.txt files were extracted. Their sizes were verified against the archive
  listing; three files truncated by an interrupted extraction were re-extracted.
* Features: 270 cases (90 per system) × 3 windows, no duplicates. Code hashes match the protocol and A1.
* Amendment A1 (source id = container_name when pod_name is absent) was made before any RE2 feature was
  computed. RE3 features were checked unchanged under A1.

## Gate (pre-declared: within-system GBDT AUROC of COUNT or UNIT ≥ 0.65)
| System | COUNT | UNIT | Passes |
|---|---|---|---|
| Online Boutique | 0.556 | 0.556 | no |
| Sock Shop | 0.511 | 0.501 | no |
| Train Ticket | 0.463 | 0.563 | no |

No system passes, so **H-W, H-WB, H-A and H-C cannot be tested**. The evaluator now handles the pre-declared
empty analysis set as the normal terminal status `NOT_TESTABLE` (the original run raised `KeyError` only
after recording the gate). This technical correction does not change the protocol or any result.
**Decision under the frozen rule:** R12 is reported as it is (not testable); no GO.

## Why (descriptive)
RE2 injects resource and network faults (CPU, MEM, DISK, SOCKET, DELAY, LOSS). Error-like log lines barely
react to them. Median error lines in the fault window equal those in the normal window in every system
(OB 0 vs 0, SS 737 vs 744, TT 10 vs 11). The error-line count changes by more than 50 % in only 18 % (OB),
7 % (SS) and 30 % (TT) of cases. For the frozen features and classifiers, the extracted error-like log
channel did not yield useful fault discrimination for this fault family, so the test is uninformative. This
does not establish that all RE2 logs are information-free. It is **neither support for nor evidence against**
the hypothesis.

## What is not done (to avoid rescuing the result)
No switch to metrics or traces, no use of RE2's level/error columns, no new windows or gate threshold, and
no other dataset picked after seeing this outcome. Any further confirmatory attempt would need a new
protocol, with the dataset chosen by a criterion fixed before looking (e.g. code-level faults with logs),
and is a decision for the author and the reviewer.

## Status of the natural-duplication evidence
RCAEval RE3 (code-level faults) remains **development / exploratory** evidence only. Sock Shop's background
copy storm masks faults in raw counts (raw count rises in 7 % of cases, distinct units in 73 %). On RE3 the
worst-case cross-system Brier favoured distinct units (+0.239 [0.059, 0.355]). Those numbers were seen while
the analysis was being designed and cannot be presented as confirmation.
