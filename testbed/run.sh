#!/usr/bin/env bash
# Host-side helper (macOS Terminal). Run from the testbed/ folder.
#   ./run.sh build | up | provision | preflight | pilot | campaign [N] | extract | status | down | reset
#   ./run.sh r5 <block> [N]   one independent R5 block (restart + provision + preflight + N episodes/class)
#   ./run.sh r5all [N]        blocks 1, 2, 3 in sequence (resumable), then extract
#   ./run.sh r10all [N]       R10: six new independent blocks 13..18, N=10 episodes/class (resumable), then extract
#   ./run.sh r6fix [N]        R6 replacement blocks 10..12 (blocks 7-9 invalid: IMSIs beyond provisioned range)
#   ./run.sh r6all [N]        R6: six new independent blocks 4..9, N=10 episodes/class (resumable), then extract
set -euo pipefail
cd "$(dirname "$0")"
DC="docker compose"
NPROV=25000   # subscribers provisioned per block; IMSI of episode idx = base + 200 + 20*idx (+0..19)
r5_block() {
  local b=$1 n=${2:-20}
  local st="out/r5_block${b}_state.json"
  if [ -f "$st" ] && grep -q '"complete": true' "$st"; then echo "block $b already complete"; return 0; fi
  echo "=== R5 block $b: restarting the core ==="
  $DC down >/dev/null 2>&1 || true
  mkdir -p out; $DC up -d; sleep 20
  local E; E=$(ls out/episodes 2>/dev/null | wc -l | tr -d ' ')
  local need=$((200 + 20 * (E + 5 * n + 5)))
  if [ "$need" -gt "$NPROV" ]; then echo "ABORT: episodes would use IMSIs beyond the $NPROV provisioned subscribers (need $need)."; exit 4; fi
  $DC exec -T mongo mongosh mongodb://localhost/open5gs --quiet --eval "var N=$NPROV" /scripts/provision.js
  $DC exec -T core python3 /scripts/campaign.py preflight
  $DC exec -T core python3 /scripts/campaign.py r5block "$b" "$n" | tee -a out/r5.log
}
case "${1:-help}" in
  build)     $DC build ;;
  up)        mkdir -p out; $DC up -d; sleep 15; $DC ps ;;
  provision) $DC exec -T mongo mongosh mongodb://localhost/open5gs --quiet --eval "var N=${2:-$NPROV}" /scripts/provision.js ;;
  preflight) $DC exec -T core python3 /scripts/campaign.py preflight ;;
  pilot)     $DC exec -T core python3 /scripts/campaign.py pilot ;;
  campaign)  $DC exec -T core python3 /scripts/campaign.py campaign "${2:-30}" | tee -a out/campaign.log ;;
  r5)        r5_block "$2" "${3:-20}" ;;
  r5all)     for b in 1 2 3; do r5_block "$b" "${2:-20}"; done
             $DC exec -T core python3 /scripts/extract.py; $DC down; echo "R5 ALL DONE" ;;
  r6all)     for b in 4 5 6 7 8 9; do r5_block "$b" "${2:-10}"; done
             $DC exec -T core python3 /scripts/extract.py; $DC down; echo "R6 ALL DONE" ;;
  r6fix)     for b in 10 11 12; do r5_block "$b" "${2:-10}"; done
             $DC exec -T core python3 /scripts/extract.py; $DC down; echo "R6 FIX DONE" ;;
  r10all)    for b in 13 14 15 16 17 18; do r5_block "$b" "${2:-10}"; done
             $DC exec -T core python3 /scripts/extract.py; $DC down; echo "R10 ALL DONE" ;;
  extract)   $DC exec -T core python3 /scripts/extract.py ;;
  status)    ls out/episodes 2>/dev/null | wc -l | xargs echo "episodes:"
             for f in out/r5_block*_state.json; do [ -f "$f" ] && echo "$f: $(cat "$f")"; done; tail -2 out/r5.log 2>/dev/null || true ;;
  down)      $DC down ;;
  reset)     $DC down; rm -rf out ;;
  *) sed -n '2,6p' "$0" ;;
esac
