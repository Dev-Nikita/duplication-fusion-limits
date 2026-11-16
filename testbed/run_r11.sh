#!/usr/bin/env bash
# R11 helper (macOS Terminal). Run from the testbed/ folder. R10 files (run.sh, campaign.py) are unchanged.
#   ./run_r11.sh loadpilot            technical pilot of the load manipulation (~25 min), then extract
#   ./run_r11.sh r11all <W> [N]       confirmatory blocks 21..32, N=10 episodes/class/condition (resumable)
#   ./run_r11.sh status
set -euo pipefail
cd "$(dirname "$0")"
DC="docker compose -f docker-compose.yml -f docker-compose.r11.yml"
NPROV=25000
start_core() {
  $DC down >/dev/null 2>&1 || true
  mkdir -p out; echo 0 > out/load_ctl
  $DC up -d; sleep 20
  $DC exec -T mongo mongosh mongodb://localhost/open5gs --quiet --eval "var N=$NPROV" /scripts/provision.js
  $DC exec -T core python3 /scripts/campaign.py preflight
}
hoststats() {   # manipulation check from the Docker side, every 5 s
  while true; do echo "T $(date +%s)"; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'; sleep 5; done >> "out/$1" 2>/dev/null &
  echo $!
}
r11_block() {
  local b=$1 W=$2 n=${3:-10} st="out/r11_block${b}_state.json"
  if [ -f "$st" ] && grep -q '"complete": true' "$st"; then echo "block $b already complete"; return 0; fi
  echo "=== R11 block $b: restarting the core ==="
  start_core
  local hp; hp=$(hoststats "r11_hoststats_b${b}.log")
  $DC exec -T core python3 /scripts/campaign_r11.py r11block "$b" "$n" "$W" | tee -a out/r11.log
  kill "$hp" 2>/dev/null || true
}
case "${1:-help}" in
  loadpilot) start_core; hp=$(hoststats r11_pilot_hoststats.log)
             $DC exec -T core python3 /scripts/campaign_r11.py loadpilot | tee -a out/r11_pilot.log
             kill "$hp" 2>/dev/null || true
             $DC exec -T core python3 /scripts/extract.py; $DC down; echo "R11 LOADPILOT DONE" ;;
  r11all)    for b in 21 22 23 24 25 26 27 28 29 30 31 32; do r11_block "$b" "$2" "${3:-10}"; done
             $DC exec -T core python3 /scripts/extract.py; $DC down; echo "R11 ALL DONE" ;;
  status)    ls out/episodes 2>/dev/null | wc -l | xargs echo "episodes:"
             for f in out/r11_*state.json; do [ -f "$f" ] && echo "$f: $(cat "$f")"; done; tail -2 out/r11.log 2>/dev/null || true ;;
  *) sed -n '2,5p' "$0" ;;
esac
