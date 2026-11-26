#!/usr/bin/env bash
# Local integration test without Docker: every component as a process on localhost (short episodes).
# Usage: bash local_test.sh <workdir> [episodes] [ep_sec]
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"; W="$1"; N="${2:-30}"; EP="${3:-2}"; M=6
[ -e "$W" ] && { echo "$W exists"; exit 1; }
mkdir -p "$W"/{phaseA,truth,phaseB}; pids=()
trap 'kill ${pids[@]} 2>/dev/null || true' EXIT
for i in $(seq 1 $M); do
  NAME=r$i LISTEN=127.0.0.1:$((18080+i)) "$B/bin/replica" & pids+=($!)
  NAME=s$i SEED=local TARGET=http://127.0.0.1:$((18080+i)) LISTEN=127.0.0.1:$((18180+i)) CONTROL=127.0.0.1:$((19090+i)) "$B/bin/faultproxy" & pids+=($!)
done
sleep 1
T0=$(( $(date +%s%3N) + 2000 )); export T0_UNIX_MS=$T0 EP_SEC=$EP EPISODES=$N
PROX=$(for i in $(seq 1 $M); do printf "s$i=http://127.0.0.1:$((19090+i)),"; done); PROX=${PROX%,}
SEED="local-$(basename "$W")" CONFIG="$B/config/faults_pilot.json" PROXIES="$PROX" TRUTH="$W/truth" "$B/bin/injector" & inj=$!
for i in $(seq 1 $M); do
  SOURCE=s$i TARGET=http://127.0.0.1:$((18180+i))/work OUT="$W/phaseA" RATE_HZ=20 TIMEOUT_MS=500 GUARD_SEC=0.3 "$B/bin/probe" & pids+=($!); pr+=($!)
done
wait $inj; for p in "${pr[@]}"; do wait $p; done
echo "phase A done"
