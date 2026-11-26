#!/usr/bin/env bash
# Runs physical runs (Phase A) and their broker replays (Phase B) on the server, one after another.
# Touches nothing outside this folder and its own Compose projects (paef_mt_a, paef_mt_b). Never overwrites a run.
# Usage: bash run_campaign.sh <pilot|main> <first_run_index> <last_run_index>
#   pilot: EPISODES=300, faults config/faults_pilot.json        main: EPISODES=400, config/faults_main.json (frozen)
set -euo pipefail
cd "$(dirname "$0")"
KIND="$1"; FIRST="$2"; LAST="$3"; EP_SEC=6
case "$KIND" in pilot) EPISODES=300; FAULTS=config/faults_pilot.json;; main) EPISODES=400; FAULTS=config/faults_main.json;; *) exit 2;; esac
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker compose version >/dev/null || { echo "docker compose plugin not found"; exit 1; }
sha256sum -c SHA256SUMS
grep -q '"note"' "$FAULTS" && { echo "$FAULTS is not frozen yet"; exit 1; }
docker compose -f compose.phaseA.yml build
for i in $(seq "$FIRST" "$LAST"); do
  RUN_ID="$KIND$i"; RUN_DIR="$PWD/runs/$RUN_ID"
  [ -e "$RUN_DIR" ] && { echo "$RUN_DIR exists: a recorded run is never overwritten"; exit 1; }
  mkdir -p "$RUN_DIR"/{phaseA,truth,phaseB}
  export RUN_DIR EP_SEC EPISODES SEED="paef-mt-$RUN_ID" FAULTS="$PWD/$FAULTS"
  export T0_UNIX_MS=$(( ($(date +%s) + 45) * 1000 ))
  { date -u +%FT%TZ; uname -srm; docker --version; docker compose version; nproc; free -h | head -2; } > "$RUN_DIR/environment.txt" 2>&1 || true
  docker compose -f compose.phaseA.yml up -d
  ids=$(docker compose -f compose.phaseA.yml ps -q injector probe1 probe2 probe3 probe4 probe5 probe6)
  docker wait $ids > "$RUN_DIR/exit_codes.txt"
  docker compose -f compose.phaseA.yml logs --no-color > "$RUN_DIR/phaseA_logs.txt" 2>&1 || true
  docker compose -f compose.phaseA.yml down -v --remove-orphans
  grep -qv '^0$' "$RUN_DIR/exit_codes.txt" && { echo "a Phase-A container failed, see $RUN_DIR"; exit 1; }
  python3 tools/make_reports.py "$RUN_DIR" "$RUN_ID" config/thresholds.json
  docker compose -f compose.phaseB.yml up --abort-on-container-exit --exit-code-from replayer
  docker compose -f compose.phaseB.yml down -v --remove-orphans
  docker image inspect nats:2.10.22-alpine --format 'nats image {{index .RepoDigests 0}}' >> "$RUN_DIR/environment.txt" 2>&1 || true
  python3 tools/audit_phaseb.py "$RUN_DIR" 2 | tee "$RUN_DIR/audit.txt"
  (cd runs && tar czf "$RUN_ID.tgz" "$RUN_ID" && sha256sum "$RUN_ID.tgz" | tee -a SHA256SUMS.runs)
done
echo "done: copy runs/*.tgz and runs/SHA256SUMS.runs back"
