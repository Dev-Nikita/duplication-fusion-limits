# R4 — confirmation of A4 on an own Open5GS + UERANSIM testbed (frozen 2026-10-17, before any data exist)

## Testbed (testbed/ in this repository)
Open5GS v2.7.6 (5G SA core, all NFs on loopback addresses, built from source) + UERANSIM v3.2.7 (gNB + UEs)
+ MongoDB 6.0, Docker Compose on the author's MacBook (Apple silicon, Docker Desktop).
Episode = 20 UEs registering over 20 s (+ 2 s drain, 5 s cool-down); one fault class per episode:
* healthy;
* amf_stall: AMF duty-cycled SIGSTOP/SIGCONT, period 0.4 s, duty U(0.05, 0.40);
* ausf_stall: same on AUSF + UDM;
* reg_reject: a fraction U(0.05, 0.5) of UEs use never-provisioned IMSIs.
Campaign: 30 episodes per class, shuffled with seed 20261023 (120 episodes). Pilot: 2 per class, seed 7 —
**pilot episodes are used only to finalise parsers and are excluded from all analyses**.
Recorded per episode: pcap (NGAP/SCTP + SBI HTTP/2), Prometheus scrapes (0.5 s), NF log slices, UE logs,
fault schedule with exact times (meta.json).

## Observation layer (naturally heterogeneous sources; lineage logged at construction)
Origins: AMF processing, AUSF/UDM authentication path, registration outcome.
Readings of an origin come from *different sources describing the same procedures*: signalling latency
or reject counts (pcap), NF log events, Prometheus counters, UE-side probe outcome. Alarm thresholds =
90th percentile of healthy TRAIN episodes. Copies = monitoring re-emissions with controlled D.
Exact parsers are fixed on pilot data and committed before campaign data are analysed.

## Analysis
Three binary tasks (each fault class vs healthy). Splits by episode: 50% train / 20% calibration / 30%
test, 5 seeds. Methods (unchanged code paths): stacking (D_train = 1), A4 lineage-aware stacking,
generative lineage model, naive Bayes. Grids: D ∈ {0, 1, 2, 4, 8}, h ∈ {0, 0.5, 1}.
**Learnability gate:** stacking AUROC on calibration episodes ≥ 0.65 per task and seed; tasks failing
in ≥ 3 of 5 seeds are excluded and reported.
Confirmatory hypotheses (same as R3): C1 at D = 8, h = 0.5 stacking − A4 Brier > 0 (paired bootstrap CI
over seeds × tasks excludes 0); C2 at D = 1, h = 0.5 A4 − stacking < 0.01 (upper CI bound); C3 A4 with
h = 0 changes < 0.005 between D = 0 and 8; C4 certified-decision error < overall A4 error at h = 0.5, D = 8.
A4 is called confirmed only if C1 and C2 hold on gated tasks.


## Parser freeze (recorded 2026-10-17, after reading ONLY pilot episodes ep_0000–ep_0007)
`experiments/r4/build_features.py`, sha256 ad9a7151d7b8b3cd691446a2b21b24e4cef3c9d020ff953c314275582b214382.
Origins → readings (source): REG → reg_sig_t90 (pcap), reg_log_p90 (AMF log), reg_ue_p90 (UE log),
reg_met_failfrac (Prometheus); AMF → amf_sig_react_p90 (pcap), amf_met_scrape_err (Prometheus),
amf_log_maxgap (AMF log); AUTH → auth_sbi_ausf_p90, auth_sbi_udm_p90, auth_sbi_udr_p90 (pcap SBI);
REJ → rej_sig (pcap), rej_log (AMF log), rej_ue (UE log), rej_sbi404 (pcap), rej_met (Prometheus).
Primary reading of each origin = the first listed; the others may have hidden lineage.
**Pilot observation recorded before the campaign was read:** stall faults at the pilot intensities
(0.06–0.23) were not visible in any reading (per-UE registration p90 0.27–0.30 s in all classes), while
reg_reject was trivially visible. The learnability gate is expected to exclude weak stall tasks.
