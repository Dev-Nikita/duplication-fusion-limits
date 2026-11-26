# K=2 whole-report experiment (Phases A and B)

Phase C (analysis): `tools/whole_report_eval/phasec.py`. The experiment protocol is frozen with the code before the
main campaign.

| Component | Role |
|---|---|
| `cmd/replica` | small HTTP service (fixed CPU work per request) |
| `cmd/faultproxy` | reverse proxy before each replica; delay, HTTP 503, resets, stall; control port for the injector only |
| `cmd/injector` | seeded schedule of classes, faults and structural conditions (nested, mixed, opposing); ground truth to its own volume |
| `cmd/probe` | one source per replica; one raw record per episode, no label |
| `tools/make_reports.py` | raw records -> frozen whole reports (thresholds in `config/thresholds.json`) |
| `cmd/replayer` | Phase B: replay through NATS JetStream, regimes S0-S6, every delivery logged |
| `tools/audit_phaseb.py` | per-run audit (standard library only) |

## Server (pilot)

Isolated Compose projects `paef_mt_a` and `paef_mt_b`: internal networks, no published ports, memory limits,
memory-only JetStream. Nothing else on the host is touched; never `docker system prune`.

```bash
cd experiments/multitype
bash run_campaign.sh pilot 1 3          # about 2 h; writes runs/pilot1..3 and their .tgz
```

Copy `runs/*.tgz` and `runs/SHA256SUMS.runs` back to the Mac, unpack under `experiments/multitype/runs/`, then

```bash
python3 tools/whole_report_eval/phasec.py gate results/multitype/pilot experiments/multitype/runs/pilot1 experiments/multitype/runs/pilot2 experiments/multitype/runs/pilot3
```

## Main campaign (only after the gate and freezing)

`config/faults_main.json` = the pilot configuration plus at most one documented fix; `design.json` fixes training and
test runs; then `bash run_campaign.sh main 1 8` and `phasec.py eval`.

## Local test without Docker

`bash local_campaign.sh <workdir> 6 40 1.5` builds nothing; it needs `bin/` (`go build -o bin/ ./cmd/...`) and a
`nats-server` binary in `bin/` listening on 127.0.0.1:14222.
