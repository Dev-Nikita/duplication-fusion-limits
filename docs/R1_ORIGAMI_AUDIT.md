# R1 — ORIGAMI audit results (2026-10-17)

## Files (all present, sizes match Zenodo record 10.5281/zenodo.19605370)
Moved from the project root to `data/origami/raw/` (git-ignored): README (3.1 kB), example.csv,
example.pcap.gz, case_study_csv.zip, amf_stability_csv.zip, case_study_ues_csv.zip, case_study.zip,
amf_stability.zip. `case_study_csv` unpacked to `data/origami/csv/` (2.0 GB).
The authors' parser lives in `5G-Signaling-Parser/` (third-party, MIT-licensed per its repo; git-ignored;
not part of our code).

## Structure (state-machine-filtered CSV; `node;servicetime;id;run;start;end;ue`)
* Case study: 16 configurations = {1, 2, 4, 8} cores × {10, 20, 50, 100} UE/s arrival, 19–20 runs each
  (316 runs), 999 UEs per run, ~1.34–1.46 M service events per configuration.
* Stability study (not yet unpacked): 6 files, 8 cores, 10 000 UEs, arrival 10…200 UE/s, 10 runs each.
* Nodes: amf, ausf, udm, udr, smf, upf, pcf, bsf, gnB (+ `none`).

## First findings (results/r1/audit_case_study.csv; experiments/r1/audit_origami.py)
* Resource starvation is visible but **not trivially**: run-level AUROC of per-run p95 service time,
  1 core vs 8 cores, AMF 0.96 / 0.99 / 0.80 / 0.80 and SMF 0.99 / 0.97 / 0.91 / 0.78 at 10 / 20 / 50 /
  100 UE/s; 2 vs 4 cores is much harder (0.32–0.82). Window-level detection will be harder still.
* Sources differ in informativeness: UDM and gNB carry little or inverted signal (AUROC 0.23–0.79).
* Real non-monotone effects exist (e.g. 4 cores at 50 UE/s shows the largest AMF/UDR p95 and flow p95).
* **Natural dependence by construction of the call flow**: a registration traverses gNB → AMF → AUSF →
  UDM → UDR (and SMF/UPF/PCF for PDU sessions). Per-NF statistics over the same window are computed
  from the *same UE procedures*, so they are dependent readings of a shared origin; a per-UE flow-duration
  statistic is a deterministic function that *contains* those service times; monitoring re-emissions are
  copies. This is exactly the typed lineage of the paper, and it can be logged exactly (message sets).

## Decisions for R2 (to be frozen in docs/r2-protocol.md)
* Unit of diagnosis: a time window of W seconds inside one run.
* Label H: starved (1 core) vs healthy (8 cores) at the same arrival rate; 2/4 cores as a shift test.
  Label is a property of the run configuration (documented as such).
* Observations: per-NF tail-latency alarms, per-window flow-duration alarm, failure/abandon counters,
  monitoring re-emissions; thresholds from calibration runs only. Every observation logs the message
  set it was computed from → lineage graph by construction (copy / overlapping reading / disjoint).
* Splits strictly by run: train (lineage scorer, thresholds) / calibration (T1 quantile) / test.
* This is semi-synthetic *in provenance* (real signalling, constructed observation layer) and must be
  called so; the own Open5GS testbed (R4) supplies naturally heterogeneous logs + metrics + signalling.
