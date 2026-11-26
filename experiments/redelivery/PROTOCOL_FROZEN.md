# Redelivery experiment — frozen protocol (2026-11-16T17:39:44Z); written before the confirmatory run

**Kind of evidence.** Controlled real transport: positive reports are replayed through NATS JetStream and copies arise
only from broker redelivery on acknowledgement timeout. It is not a production data set, it reuses the R6 and R10
testbed readings (new transport on old measurements), and it does not prove a minimax bound.

**Frozen files (sha256).**
```
69f89ef5b7f3d3d168093727bf8096608218b4f33f05b5db759458f5e5d5781f  main.go
1937451ba50eb06f0982efdb130d9acc9b78a6de7eb544d94b16540791df8890  go.mod
cbc12a2f23784c6f436029dcfe3cda80c1b59f8e7c008354ee869356b733dbf4  go.sum
fd6421917ef34ab60f0e883531f7394cb5dfaf9436b45918a60117d0834c95ab  Dockerfile
d17670aec3db1341005c6b945ffc91221e64baa0ac2b49846b87bd2a32e5c7b6  docker-compose.yml
68fa2f5551a712176fa1b7226023381fd81c370b6805be735e728f2fa442c57a  input/reports.jsonl
5d88b062f9b9ea00c0d0c1a3ffb17d76bc934592b00b488b6da9c79c8cfda047  prepare.py
bc3f6378e1ca654ae3dcc02f55bf86e5c801f5b714911e8879a6b6827272a11c  evaluate.py
c0dff994eea952dd5ec669aa16ac8d4d6d29fc4b874d7c70a2b193d6f6281c4b  input/truth.csv
3517932d8767a38bb0de45bd160c4c450a95a70fe3bdb6953303418b7552a1d9  tools/finite_cap_cvx.py
e9f797949a1e714f408e7648172e0a0d5d83f120f54d4ad2821de4220e268a85  tools/frontier.py
```
Seed of the confirmatory run: `confirm-20261116`. AckWait 250 ms. One run on the Linux server; `run.sh` refuses to overwrite an existing output.

**Signals and labels.** Episodes of R6 (450) and R10 (300), 20 UEs each. A reading fires if its UE-side latency exceeds the
90th percentile of the finite healthy readings of the training blocks of its campaign; a timed-out reading fires. y = 1 for a
fault episode (any of four fault types), 0 for healthy. Split by physical block: the last third of the blocks of each
campaign is held out (R6: 150 episodes, R10: 100). The producer file holds no label, block or split; the auditor file
`truth.csv` never leaves the analysis machine.

**Regimes.** A report is acknowledged when its processing ends; the broker redelivers it every AckWait until then, at most
MaxDeliver times. S0: processing never exceeds AckWait. S1 (stable): 30% of reports are slow (1.1–2.5 AckWait), MaxDeliver 3.
S2 (bursts): 20% of episodes are bursts with 90% slow reports, 5% elsewhere. S3a (shift within the cap): 70% slow,
MaxDeliver 3. S3b (shift beyond the declared cap): 70% slow up to 4.5 AckWait, MaxDeliver 5; reported as an
outside-model stress test. S4 (content-coupled): processing time grows with the reported severity,
min(2.5, 2(severity−1)) AckWait, so the copying depends on the reading and through it on the state, without access to the label.
Which reports are slow is fixed by a hash of the seed and the report identifier; how many deliveries result is decided by
the broker's timers.

**Model check (from the trace, before any reading of results).** Every delivery maps to one root and one stream sequence;
delivery counters are 1..n; multiplicity at most MaxDeliver; every original delivered; no copies in S0 and some in all
other regimes. Any violation: the run is reported as failed, not repaired.

**Rules.** All see only the count A at test time. Fitted on the training blocks under S1: presence table; count table;
count GBDT (depth 3, 200 iterations, rate 0.1); clipped count; log count; plug-in minimax with declared cap D = 2 (law of
(y,U) from the training blocks, which needs lineage in training only); weighted Bayes–minimax with ε = 0.2 (fixed, not
tuned). Reference, not a competitor: table on the unique count U.

**Outcomes.** Held-out Brier per campaign and regime (observed). Separately: model worst case over Δ_2 of the minimax
and weighted rules on the held-out law. Costs: bytes per delivery and deliveries per report from the trace.

**Pre-specified readings (per campaign; margin 0.010 Brier; each may fail).**
H1: over the within-cap regimes S1, S2, S3a, S4, the better of minimax and weighted has a worst observed Brier lower than
that of the count table by at least 0.010.
H2: under S1 the count table is not worse than minimax by more than 0.010.
Learnability gate: AUROC of U on the training blocks at least 0.65, else the campaign is reported as inconclusive.
No intervals across the five held-out blocks are claimed; per-regime values are descriptive.

**Not in this run (declared now).** Two report types with whole-report messages; a second, independent service as signal
source; negative reports; measurement of CPU and latency of the fuser. They are not claimed.

**Engineering pilot (recorded).** Run locally on the first two training blocks of each campaign (200 episodes, seed
`pilot`), nats-server 2.10.22. Used only to check delivery, the audit and the code. It showed that with the 90th
percentile of all healthy readings the R6 threshold is infinite (45% of healthy R6 readings time out), so nothing fired;
the threshold was changed to the 90th percentile of the finite healthy readings, on training blocks only. No held-out
block was looked at.
