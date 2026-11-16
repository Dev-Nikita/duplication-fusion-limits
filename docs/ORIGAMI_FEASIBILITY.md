# ORIGAMI 5G Signaling Traffic — data feasibility audit (v0, 2026-10-17)

## Access
* Zenodo record 10.5281/zenodo.19605370 (CC BY 4.0, ~7 GB). Neither the analysis container nor the
  linked VM can download from Zenodo (egress proxy 403); the README fetch was rate-limited (HTTP 429).
  **Download on the Mac by hand**: `case_study_csv.zip` (378 MB), `amf_stability_csv.zip` (550 MB),
  `example.csv` (125 MB), `README.md`; raw PCAPs only if a field is missing from the CSVs.
  Put them in `data/origami/raw/` (git-ignored).

## What the authors' parser provides (github.com/lsinfo3/5G-Signaling-Parser, read 2026-10-17)
* `dialogparser`: parses HTTP/2 (SBI), NGAP and PFCP from PCAP; tracks a UE across RAN-UE-NGAP-ID,
  AMF-UE-NGAP-ID, SUCI, SUPI, PDU session ID, SEID, SM context; groups messages into *dialogs* by protocol
  state machines; exports per-UE CSVs (`;`-separated).
* `signaling_filter.py`: checks each per-UE sequence against a state machine, aggregates messages into
  processes, computes per-NF **service times**, exports `node;servicetime;id;run;start;end;ue`.
* Experiments (Zenodo description): case study with 1,000 UEs and 1/2/4/8 CPU cores per NF VM; AMF
  stability study with 10,000 UEs, 8 cores, arrival rates incl. 150 and 200 UE/s.

## Consequences for PAEF (reviewer points accepted)
1. IMSI / NGAP IDs / message IDs give **sequences and dialogs**, not information lineage. They do not
   prove that two observations are copies or independent.
2. Lineage will therefore be **defined by construction of our own observation layer**: every diagnostic
   observation (per-window NF service-time alarm, per-UE procedure-failure flag, per-NF failure counter,
   re-emitted alert) is computed by our code from an explicit set of parsed messages, and the code logs
   that set. Two observations are
   * copies if one is a re-emission of the other (same message set, same value),
   * readings of one origin if their message sets overlap (shared source messages),
   * independent if their message sets are disjoint and come from different UEs / NFs / windows.
   This makes the answer key exact *for our pipeline*; the paper must say so and must not claim lineage
   "in the dataset".
3. No fault labels exist. Candidate incident label: **resource-starvation regime** (1 vs 8 cores per NF)
   or **overload regime** (arrival rate) — taken from the experiment configuration, i.e. a property of
   the run, not of each message. Whether per-window service times separate these regimes is the first
   thing to check (if they separate perfectly, the task is too easy to be informative).
4. Lineage queries (G3) must be realistic: "check whether observation j shares source messages with
   origin r" ⇔ look up the message-ID set — available in our pipeline; cost = one lookup.

## Checklist before any experiment (fill after download)
- [ ] File list and sizes match Zenodo; licence noted.
- [ ] Columns of `*_m_stats.csv` and filtered CSVs documented.
- [ ] Runs identifiable; mapping run → (cores, arrival rate) recovered from file names / README.
- [ ] Per-NF service-time distributions per regime plotted; separability checked.
- [ ] Observation layer v0 (5 observation types) with message-set logging; lineage graph derived.
- [ ] Split by run (calibration runs vs test runs), never by message.
