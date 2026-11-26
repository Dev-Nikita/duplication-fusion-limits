# Real-redelivery experiment

`PROTOCOL_FROZEN.md` (+ `.sha256`) is the protocol; it was frozen before the confirmatory run.
`main.go` replays positive reports through NATS JetStream and logs every delivery; copies come from broker redelivery only.
`prepare.py` builds the producer input (`input/reports.jsonl`, no labels) and the auditor file (`input/truth.csv`).
`evaluate.py` audits the lineage and evaluates the rules. `bundle/` is what goes to the server (no auditor file).
`pilot_*` hold the engineering pilot on training blocks (not a result).

After the server run:  tar xzf paef_redelivery_out.tgz -C confirm/  and
  python3 -W ignore evaluate.py confirm/out input/truth.csv ../../tools ../../results/redelivery
