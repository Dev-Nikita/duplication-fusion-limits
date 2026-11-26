# Reproduction targets. Run from the project root. See REPRODUCE.md.
export OMP_NUM_THREADS=1
PY=python3
FLIGHT_DATA?=data_external/lunadong_fusion/clean_flight
FLIGHT_TRUTH?=data_external/lunadong_fusion/flight_truth

.PHONY: integrity whole-report-eval whole-report-quick whole-report-full redelivery-corrected frontier cross-platform reproduce-math reproduce-tables reproduce-figures reproduce-external clean-room audit manifest versions

versions:
	$(PY) -c "import sys,numpy,pandas,scipy,sklearn,cvxpy,matplotlib;print(sys.version.split()[0],numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__,cvxpy.__version__,matplotlib.__version__)"

reproduce-math:
	$(PY) tools/test_finite_cap_independent.py | tail -1
	$(PY) tools/saturation_closed_form.py | tail -1
	$(PY) -W ignore tools/quantitative_saturation.py | tail -1
	$(PY) -W ignore tools/extended_models.py | tail -1
	$(PY) -m pytest -q tests -p no:cacheprovider -W ignore | tail -1

reproduce-tables:
	$(PY) tools/summarise_finite_cap.py > /dev/null
	$(PY) tools/make_table4.py > /dev/null
	$(PY) tools/make_stability_table.py > /dev/null
	$(PY) tools/make_baseline_table.py > /dev/null
	@echo "written: results/finite_cap/table4_rows.tex, stability_rows.tex, results/est/baseline_rows.tex"

reproduce-figures:
	$(PY) tools/summarise_finite_cap.py > /dev/null
	$(PY) -W ignore tools/paper_figures.py
	$(PY) experiments/paper/figure1_values.py

reproduce-external:
	$(PY) -I -W ignore experiments/flight/flight_study.py $(FLIGHT_DATA) $(FLIGHT_TRUTH) results/flight
	$(PY) -I experiments/flight/flight_evaluate.py results/flight > results/flight/evaluate.txt
	@head -8 results/flight/evaluate.txt
	$(PY) -I -W ignore experiments/flight/flight_audit.py $(FLIGHT_DATA) $(FLIGHT_TRUTH) results/flight/audit_R2 > results/flight/audit_R2.txt
	$(PY) -I -W ignore experiments/flight/flight_audit_descriptors.py $(FLIGHT_DATA) $(FLIGHT_TRUTH) results/flight/audit_R2

# Data-integrity gate: block validity registry and checks (results/integrity/).
integrity:
	$(PY) tools/data_integrity_audit.py

# Whole-report copying (Proposition 3(i), Theorem 3, Proposition 8; Supplementary S8). Exact rational arithmetic.
WR=../../tools/whole_report_saturation
whole-report-quick:
	cd results/whole_report && $(PY) $(WR)/finite_cap.py k2form && $(PY) $(WR)/finite_cap.py k1prop4 && $(PY) $(WR)/finite_cap.py edges > /dev/null && $(PY) $(WR)/signature_cells.py && $(PY) $(WR)/kkt_verify.py out/certs/cfg*.jsonl
	$(PY) -m pytest -q tests/test_whole_report_saturation.py -p no:cacheprovider | tail -1

# Several minutes to about an hour on a laptop; certificates and gap checks are written to results/whole_report/out.
whole-report-full: whole-report-quick
	cd results/whole_report && $(PY) $(WR)/finite_cap.py laws > out/laws.log && $(PY) -W ignore $(WR)/finite_cap.py numeric && $(PY) -W ignore $(WR)/finite_cap.py k3axis && $(PY) $(WR)/axis_rule_k3.py
	cd results/whole_report && for i in 0 1 2 3 4 5 6 7; do $(PY) -W ignore $(WR)/cert_audit.py $$i && $(PY) $(WR)/quant_gap.py $$i; done
	cd results/whole_report && $(PY) $(WR)/kkt_verify.py out/certs/cfg*.jsonl

# Broker replay without the invalid R6 blocks 7-9 (Section 8.4 and S7 of the article; correction of 24 Nov 2026).
RC=../../results/redelivery_corrected
redelivery-corrected:
	cd experiments/redelivery && $(PY) correct_invalid_blocks.py confirm/out input/truth.csv ../../results/r6/units.csv $(RC)/work
	cd experiments/redelivery && for v in A B; do $(PY) -W ignore evaluate.py $(RC)/work/$$v/out $(RC)/work/$$v/truth.csv ../../tools $(RC)/$$v > /dev/null && $(PY) -W ignore robust_risk.py $(RC)/work/$$v/out $(RC)/work/$$v/truth.csv ../../tools $(RC)/$$v > /dev/null; done
	$(PY) tools/redelivery_tables.py results/redelivery_corrected/B results/redelivery_corrected/B/tables.tex
	@grep "H1\|H2" results/redelivery_corrected/B/summary.txt

# K=2 whole-report evaluation: exploratory dry run on the existing testbed (about 1 min) and the library tests.
# The K=2 experiment itself: experiments/multitype/README.md (Phases A, B) and tools/whole_report_eval/phasec.py (C).
whole-report-eval:
	$(PY) -W ignore tools/whole_report_eval/testbed_dryrun.py results/whole_report_eval
	$(PY) -m pytest -q tests/test_whole_report_eval.py -p no:cacheprovider | tail -1

frontier:
	$(PY) -W ignore tools/frontier.py testbed > results/frontier/summary_testbed.txt
	$(PY) -W ignore tools/frontier.py flight $(FLIGHT_DATA) $(FLIGHT_TRUTH) > results/frontier/summary_flight.txt

cross-platform:
	$(PY) -W ignore tools/cross_platform_check.py 0 108

clean-room:
	@test -n "$(DIR)" || (echo "usage: make clean-room DIR=<empty dir>"; exit 1)
	bash tools/clean_room_run.sh $(DIR)

manifest:
	$(PY) tools/make_manifest.py

# The three table bodies (*.tex) are generated files and are not stored in the archive; audit regenerates them first.
audit: reproduce-tables integrity
	$(PY) tools/make_manifest.py --check
	@if git rev-parse --git-dir > /dev/null 2>&1; then bash tools/git_safety_check.sh; else echo "not a git repository: hygiene check skipped"; fi

