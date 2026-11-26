#!/usr/bin/env bash
# One confirmatory run. Touches nothing outside this folder and its own Compose project (paef_redelivery).
set -euo pipefail
cd "$(dirname "$0")"
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker compose version >/dev/null || { echo "docker compose plugin not found"; exit 1; }
sha256sum -c SHA256SUMS
[ -e out ] && { echo "out/ exists: a recorded run is never overwritten. Move it away first."; exit 1; }
mkdir out
echo "free disk here: $(df -h . | tail -1 | awk '{print $4}'); free memory: $(free -h | awk '/Mem:/{print $7}')"
docker compose build
docker compose up --abort-on-container-exit --exit-code-from runner
{ date -u +%FT%TZ; uname -srm; docker --version; docker compose version; docker image inspect nats:2.10.22-alpine --format 'nats image {{index .RepoDigests 0}}'; nproc; } > out/environment.txt 2>&1 || true
docker compose down -v --remove-orphans
tar czf paef_redelivery_out.tgz out SHA256SUMS
sha256sum paef_redelivery_out.tgz
echo "done: copy paef_redelivery_out.tgz back"
