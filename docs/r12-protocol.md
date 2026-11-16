# R12 protocol — count vs distinct-unit evidence under natural duplication (RCAEval RE2)
**Frozen 2026-10-24, before RE2 was downloaded.** RE2 has not been seen by anyone in this project.
This is a test on independent public data, not a new testbed campaign. The "no R12" rule of R11 referred to
further 5G-testbed campaigns in search of a positive result; that this R12 departs from it is disclosed.

## History that must be disclosed
* Feasibility pilot on RCAEval **RE3** (counts only): natural repeated error messages, with system- and
  state-dependent multiplicity (docs/FEASIBILITY_EXTERNAL_DATA_2026-10-24.md).
* **Development on RE3** (all results seen; RE3 is not confirmatory evidence):
  1. First pipeline: GBDT with min_samples_leaf = 20 (library default) produced constant predictions in small
     folds (Sock Shop AUROC 0.50). Fixed to min_samples_leaf = 5.
  2. The first draft's primary hypothesis used the **mean** case Brier difference COUNT − UNIT. On RE3 it was
     not confirmed (mean −0.038 with the faulty GBDT). Online Boutique favours counts (fault-driven copy storms);
     Sock Shop favours units (a background copy storm masks faults).
  3. After seeing this, the primary criterion was changed to the **worst case over held-out systems**. This is
     the criterion of R10 P1 and R11-X and the quantity bounded by Theorem A. The change was made after
     inspecting development data, and is disclosed as such. The mean contrast is kept as secondary H-A.
  4. RE3 results with the frozen pipeline (development only): H-W +0.239 [0.059, 0.355]; H-WB −0.015
     [−0.042, 0.035]; H-A +0.101 [0.020, 0.175]; H-C +0.105 [−0.079, 0.264] (GBDT). Logit: all four positive.

## Data
* RCAEval **RE2** (Pham et al., arXiv 2412.17015; Zenodo record 14504481, CC-BY-4.0): 270 cases in Online
  Boutique, Sock Shop and Train Ticket. Resource and network faults (CPU, MEM, DISK, SOCKET, DELAY, LOSS) —
  a *different fault family* from RE3's code-level faults. The RE2.zip checksum is verified against the
  Zenodo-published checksum before extraction.
* Only logs.csv and inject_time.txt are used. Metrics and traces are unused. **No root-cause localisation**:
  detection only.

## Pipeline (frozen code below; nothing tuned on RE2)
* experiments/ext/rca_features.py: windows H [t0−600, t0−420], N [t0−360, t0−180], F [t0+60, t0+240]. Raw
  quantities A, U, P, T, L, K as defined in the file.
* experiments/ext/rca_model.py: feature sets COUNT {A}, UNIT {U, P, T}, BOTH {A, U, P, T}, RATE {A/L, U/K}, each
  as a log ratio to the history window. Learners: GBDT (depth 2, 100 iterations, lr 0.1, min_samples_leaf 5)
  and logistic regression. Schemes: within-system 5-fold grouped by fault group; cross-system (train on
  two systems, test on the third).
* experiments/ext/rca_evaluate.py: gate (within-system GBDT AUROC of COUNT or UNIT ≥ 0.65), hypotheses,
  and a cluster bootstrap over fault groups (B = 5000, seed 20261024).

## Hypotheses (GBDT primary; logistic reported)
* **H-W (primary):** worst-case cross-system Brier of COUNT minus that of UNIT is > 0. Bootstrap CI95 lower
  bound > 0.
* H-WB: the same for BOTH vs UNIT (do counts added to distinct units hurt under transfer?).
* H-A: mean case Brier difference COUNT − UNIT (cross-system) > 0.
* H-C: the worst-case gap is larger cross-system than within-system.
* Reported without a test: RATE vs UNIT; all AUROC and Brier values per system, scheme, learner and set.

## Decision rule
H-W confirmed → GO: the natural-duplication result enters the paper as confirmatory evidence for the
worst-case (robustness) value of distinct-unit evidence over raw counts, with RE3 as development and the 5G
testbed and simulation as supporting evidence. H-W not confirmed → R12 is reported as it is, and the paper
stays a boundary study. In either case there are no further confirmatory sets.

## Frozen code (sha256)
```
51482cb545861e88f32c03a193ac9bdb5fa3ba4485745a6de1387b5381b104d0  experiments/ext/rca_features.py
d3e4659c1a575c841d89dac20159067aa3e59809f1491955a6e993c446cddcf6  experiments/ext/rca_model.py
1e3a7a5f0ed5ab49e64a4bfbae62c214c97df2da9259d25265dd3d307b70db83  experiments/ext/rca_evaluate.py
```

## Amendment A1 (2026-10-24, after download and checksum, before any RE2 feature or outcome was computed)
RE2's logs.csv has no pod_name column (columns: time, timestamp, container_name, message, level, req_path,
error, …; seen from headers only). The source id is now pod_name if present, else container_name. In RE3
there is about one pod per service, so this is the same unit. The extra RE2 columns (level, error) are **not**
used, so the frozen error-line definition (regex on message) stays unchanged. Check that RE3 features are
unchanged by the amendment: see below. New sha256:
```
199eeb0dfe96ee5b3d84092f0964cd03f3cb90b8cadc7a388c6dca15cab7b573  experiments/ext/rca_features.py
```
