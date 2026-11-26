#!/usr/bin/env bash
# Local end-to-end test: several short physical runs, reports, broker replay of every regime.
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"; BASE="$1"; RUNS="${2:-4}"; N="${3:-40}"; EP="${4:-1.5}"
for i in $(seq 1 $RUNS); do
  bash "$B/local_test.sh" "$BASE/run$i" "$N" "$EP" >/dev/null 2>&1
  python3 "$B/tools/make_reports.py" "$BASE/run$i" "run$i" "$B/config/thresholds.json"
  NATS_URL=nats://127.0.0.1:14222 REPORTS="$BASE/run$i/reports.jsonl" OUT="$BASE/run$i/phaseB" REGIMES=S0,S1,S2,S3,S4,S5,S6 \
    SEED=local-$i ACK_WAIT_MS=250 PUB_RATE=100 LOAD_REF=20 "$B/bin/replayer" 2>/dev/null
done
echo campaign done
