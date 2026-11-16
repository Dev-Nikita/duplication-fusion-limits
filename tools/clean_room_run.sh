#!/usr/bin/env bash
# Full clean-room reproduction (two parallel lanes, about 2.5 h on 2 cores) of the finite-cap analysis from the raw per-UE units.
# Usage: bash tools/clean_room_run.sh <empty_dir>   (run from the project root; copies only what REPRODUCE lists)
set -euo pipefail
SRC="$(pwd)"; DST="$1"; mkdir -p "$DST"; export OMP_NUM_THREADS=1
cp -r --parents tools/finite_cap_solvers.py tools/finite_cap_cvx.py tools/test_finite_cap_independent.py \
  tools/saturation_closed_form.py tools/quantitative_saturation.py tools/apply_finite_cap.py tools/presence_rule_oos.py \
  tools/fold_laws.py tools/plugin_cvx_oos.py tools/li_vs_ld.py tools/li_certify.py tools/summarise_finite_cap.py \
  tools/make_table4.py tools/block_sensitivity.py tools/saturation_stability.py tools/make_stability_table.py tools/extended_models.py tools/exchangeability_check.py tools/multitype_data.py tools/make_baseline_table.py experiments/est/baselines.py experiments/est/estimation_error.py \
  experiments/paper/figure1_values.py results/r10/units.csv results/r6/units.csv results/r11/units_cond.csv \
  results/r11x/r11_cells.csv results/r11x_rep/r10_alarm/r11_cells.csv results/r11x_rep/r10_full/r11_cells.csv \
  results/r11x_rep/r6_alarm/r11_cells.csv results/r11x_rep/r6_full/r11_cells.csv "$DST"/
cd "$DST"; mkdir -p results/finite_cap results/est
S1='R10:results/r10/units.csv:13,14,15,16,17,18'; S2='R6:results/r6/units.csv:4,5,6,10,11,12'
S3='R11-control:results/r11/units_cond.csv:21,22,23,24,25,26,27,28,29,30,31,32:control'
S4='R11-loaded:results/r11/units_cond.csv:21,22,23,24,25,26,27,28,29,30,31,32:loaded'
step(){ echo "=== $(date +%T) $*"; }
step 1;  python3 tools/test_finite_cap_independent.py | tail -1
step 2;  python3 tools/saturation_closed_form.py | tail -1
step 2b; python3 -W ignore tools/quantitative_saturation.py | tail -1
# two parallel lanes (2 cores): lane B runs the two longest steps
( step 3b; python3 -W ignore tools/apply_finite_cap.py results/finite_cap/part2 "$S3" "$S4" > /dev/null
  step 12; python3 -W ignore tools/saturation_stability.py 500 > /dev/null; python3 tools/make_stability_table.py > /dev/null
  step laneB-done ) > laneB.log 2>&1 &
LANEB=$!
step 3;  python3 -W ignore tools/apply_finite_cap.py results/finite_cap/part1 "$S1" "$S2" > /dev/null
step 4;  python3 -W ignore tools/presence_rule_oos.py results/finite_cap/presence_rule_oos.csv "$S1" "$S2" "$S3" "$S4"
step 5;  python3 -W ignore tools/fold_laws.py results/finite_cap/fold_laws.pkl "$S1" "$S2" "$S3" "$S4"
step 5b; python3 tools/saturation_closed_form.py --folds results/finite_cap/fold_laws.pkl results/finite_cap/saturation_closed_form.csv
step 6;  python3 -W ignore tools/plugin_cvx_oos.py
step 7;  python3 -W ignore tools/li_vs_ld.py > /dev/null
step 7b; python3 -W ignore tools/li_certify.py > /dev/null
step 9;  python3 -W ignore experiments/est/estimation_error.py results/finite_cap/fold_laws.pkl results/est/estimation_error.csv > /dev/null
step 10; python3 -W ignore tools/block_sensitivity.py > /dev/null
step 11a; python3 experiments/paper/figure1_values.py > /dev/null
step 13; python3 -W ignore experiments/est/baselines.py results/finite_cap/fold_laws.pkl results/est/baselines.csv > /dev/null
step 14; python3 -W ignore tools/exchangeability_check.py > /dev/null; python3 -W ignore tools/multitype_data.py > /dev/null
step 4x; python3 -W ignore tools/extended_models.py | tail -1
step wait-laneB; wait $LANEB; cat laneB.log
step 8;  python3 tools/summarise_finite_cap.py > /dev/null
step 8b; python3 tools/make_table4.py > /dev/null; python3 tools/make_baseline_table.py > /dev/null
step compare
for f in $(cd "$SRC" && ls results/finite_cap/*.csv results/finite_cap/*.tex results/finite_cap/part*/*.csv results/est/estimation_error.csv results/est/baselines.csv results/est/baseline_rows.tex results/editorial/fig1_channel_gaps.csv); do
  if [ -f "$f" ]; then cmp -s "$SRC/$f" "$f" && echo "SAME $f" || echo "DIFF $f"; else echo "MISSING $f"; fi
done
step done
