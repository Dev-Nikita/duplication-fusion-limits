# Reproduction targets. Run from the project root. See REPRODUCE.md.
export OMP_NUM_THREADS=1
PY=python3
FLIGHT_DATA?=data_external/lunadong_fusion/clean_flight
FLIGHT_TRUTH?=data_external/lunadong_fusion/flight_truth

.PHONY: cross-platform reproduce-math reproduce-tables reproduce-figures reproduce-external clean-room audit manifest versions paper

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
	$(PY) experiments/paper/figure1_values.py

reproduce-external:
	$(PY) -I -W ignore experiments/flight/flight_study.py $(FLIGHT_DATA) $(FLIGHT_TRUTH) results/flight
	$(PY) -I experiments/flight/flight_evaluate.py results/flight > results/flight/evaluate.txt
	@head -8 results/flight/evaluate.txt
	$(PY) -I -W ignore experiments/flight/flight_audit.py $(FLIGHT_DATA) $(FLIGHT_TRUTH) results/flight/audit_R2 > results/flight/audit_R2.txt
	$(PY) -I -W ignore experiments/flight/flight_audit_descriptors.py $(FLIGHT_DATA) $(FLIGHT_TRUTH) results/flight/audit_R2

cross-platform:
	$(PY) -W ignore tools/cross_platform_check.py 0 108

clean-room:
	@test -n "$(DIR)" || (echo "usage: make clean-room DIR=<empty dir>"; exit 1)
	bash tools/clean_room_run.sh $(DIR)

manifest:
	$(PY) tools/make_manifest.py

audit:
	$(PY) tools/make_manifest.py --check
	@if git rev-parse --git-dir > /dev/null 2>&1; then bash tools/git_safety_check.sh; else echo "not a git repository: hygiene check skipped"; fi

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main_if.tex > /dev/null && latexmk -pdf -interaction=nonstopmode supplementary_if.tex > /dev/null && echo "paper/main_if.pdf, paper/supplementary_if.pdf"
