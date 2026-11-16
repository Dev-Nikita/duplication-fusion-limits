"""Release manifest: article item -> inputs -> command -> output files -> sha256.
Usage: python3 tools/make_manifest.py            (writes MANIFEST.csv)
       python3 tools/make_manifest.py --check    (verifies the checksums recorded in MANIFEST.csv)"""
import csv, hashlib, os, sys

ITEMS = [
 ("Theorem 2 (numerical check)", "Sec. 5", "-", "python3 tools/test_finite_cap_independent.py", ["tools/test_finite_cap_independent.py", "tools/finite_cap_solvers.py"]),
 ("Proposition 4 (numerical check)", "Sec. 5", "-", "python3 tools/saturation_closed_form.py", ["tools/saturation_closed_form.py"]),
 ("Proposition 5 (numerical check)", "Sec. 5", "-", "python3 -W ignore tools/quantitative_saturation.py", ["tools/quantitative_saturation.py", "results/finite_cap/quantitative_saturation_check.txt"]),
 ("Propositions 6-7 (numerical check)", "Sec. 5.1-5.2", "-", "python3 -W ignore tools/extended_models.py", ["tools/extended_models.py", "results/finite_cap/extended_models_check.txt"]),
 ("Edge cases", "Sec. 5", "-", "python3 -m pytest -q tests/test_theory_adversarial.py", ["tests/test_theory_adversarial.py"]),
 ("Table 3", "Sec. 7.2", "results/r10/units.csv; results/r6/units.csv", "experiments/r11x/replicate.py; experiments/r11x/evaluate_r11x.py; experiments/r11x/robust_seeds.py",
  ["results/r11x_rep/r10.json", "results/r11x_rep/r6.json", "results/r11x_rep/r10_eval.txt", "results/r11x_rep/r6_eval.txt", "results/r11x_seeds10/r10.json", "results/r11x_seeds10/r6.json"]),
 ("Table 4", "Sec. 7.3", "results/r10/units.csv; results/r6/units.csv; results/r11/units_cond.csv", "REPRODUCE.md steps 6-9",
  ["results/finite_cap/part1/finite_cap_application.csv", "results/finite_cap/part2/finite_cap_application.csv", "results/finite_cap/plugin_cvx_oos.csv",
   "results/finite_cap/presence_rule_oos.csv", "results/finite_cap/li_vs_ld.csv", "results/finite_cap/li_certificate.csv",
   "results/finite_cap/saturation_closed_form.csv", "results/finite_cap/summary.csv", "results/finite_cap/table4_rows.tex"]),
 ("Figure 1", "Sec. 7.3", "results/finite_cap/summary.csv", "python3 tools/summarise_finite_cap.py", ["results/finite_cap/summary.csv"]),
 ("Block-level and leave-one-block-out", "Sec. 7.3", "per-UE readings", "python3 -W ignore tools/block_sensitivity.py",
  ["results/finite_cap/block_full_rerun.csv", "results/finite_cap/block_level.csv", "results/finite_cap/block_jackknife.csv"]),
 ("Table 5", "Sec. 7.3", "per-UE readings", "REPRODUCE.md step 11",
  ["results/finite_cap/saturation_stability_full.csv", "results/finite_cap/saturation_stability_boot.csv", "results/finite_cap/saturation_stability_lobo.csv", "results/finite_cap/stability_rows.tex"]),
 ("Table 6", "Sec. 7.4", "results/finite_cap/fold_laws.pkl", "REPRODUCE.md steps 12-13",
  ["results/est/estimation_error.csv", "results/est/baselines.csv", "results/est/baseline_rows.tex", "docs/est-protocol.md"]),
 ("Mechanism information; second type", "Sec. 7.5", "per-UE readings", "REPRODUCE.md step 14", ["results/finite_cap/exchangeability.csv", "results/finite_cap/multitype.csv"]),
 ("Alertmanager emulation", "Sec. 7.6", "results/r6/units.csv", "experiments/r11x/explore_r8.py", ["results/r8x/r8x_cells.csv"]),
 ("Tables S7-S8 (frozen run)", "S4", "Flight archives (external)", "REPRODUCE.md step 16",
  ["docs/flight-protocol.md", "experiments/flight/flight_study.py", "experiments/flight/flight_evaluate.py", "experiments/flight/flight_io.py",
   "results/flight/flight_results.csv", "results/flight/flight_descriptors.json", "results/flight/evaluate.txt"]),
 ("Table 7 (corrected), Tables S5-S6: gold-source audit", "Sec. 8.1; S4", "Flight archives (external)", "REPRODUCE.md step 17",
  ["docs/flight-addendum-R2.md", "experiments/flight/flight_audit.py", "experiments/flight/flight_audit_descriptors.py",
   "results/flight/audit_R2/flight_gold_overlap.csv", "results/flight/audit_R2/flight_source_gold_agreement.csv",
   "results/flight/audit_R2/flight_gold_overlap_summary.json", "results/flight/audit_R2/flight_gold_exclusion_results.csv",
   "results/flight/audit_R2/flight_descriptors_no_gold_sites.json"]),
 ("Cross-platform check", "S5", "results/finite_cap/fold_laws.pkl", "python3 -W ignore tools/cross_platform_check.py 0 108",
  ["tools/cross_platform_check.py", "results/finite_cap/cross_platform_2026-11-16.txt", "requirements_lock_2026-11-16.txt"]),
 ("RCAEval RE3 / RE2", "Sec. 8.2", "RCAEval (external)", "experiments/ext/*.py", ["results/ext/re3_dev.json", "results/r12/re2_cells.csv"]),
 ("Figure S1; Table S3", "S3", "results/r11x/r11_cells.csv; results/r11x_rep/*/r11_cells.csv", "python3 experiments/paper/figure1_values.py", ["results/editorial/fig1_channel_gaps.csv"]),
 ("Table S4", "S3", "results/r11x cells", "experiments/r11x/evaluate_r11x.py", ["results/editorial/absolute_brier_ld.csv"]),
 ("Table S2 and Sec. S2", "S2", "-", "experiments/sim/run_resumable.py; experiments/sim/summarize.py", ["results/sim/sensitivity.csv", "results/sim/summary.txt", "docs/sim-protocol.md"]),
 ("Clean-room log", "S5", "per-UE readings", "bash tools/clean_room_run.sh <dir>", ["results/finite_cap/clean_room_2026-11-15.log"]),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else "MISSING"


if "--check" in sys.argv:
    bad = 0
    for r in csv.DictReader(open("MANIFEST.csv")):
        if sha(r["output_file"]) != r["sha256"]:
            bad += 1; print("MISMATCH", r["output_file"])
    print("manifest check:", "OK" if bad == 0 else f"{bad} mismatches"); sys.exit(1 if bad else 0)
rows = [(it, sec, inp, cmd, f, sha(f)) for it, sec, inp, cmd, files in ITEMS for f in files]
w = csv.writer(open("MANIFEST.csv", "w", newline="")); w.writerow(["article_item", "section", "inputs", "command", "output_file", "sha256"]); w.writerows(rows)
print(len(rows), "rows;", sum(r[5] == "MISSING" for r in rows), "missing:", [r[4] for r in rows if r[5] == "MISSING"])
