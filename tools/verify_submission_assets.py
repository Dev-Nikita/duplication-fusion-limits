#!/usr/bin/env python3
"""Mechanical submission checks for the IF manuscript package."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / "paper/main_if.tex").read_text(encoding="utf-8")
highlights = (ROOT / "paper/highlights_if.txt").read_text(encoding="utf-8").splitlines()
cover = (ROOT / "paper/cover_letter_if.txt").read_text(encoding="utf-8")

for line in highlights:
    content = line[2:] if line.startswith("- ") else line
    assert len(content) <= 81, (len(content), content)

assert "minimax fuser of Thm" not in tex
assert "None of these shares data, results or text" not in cover

name_map = {"storm_inc": "incident storm", "storm_ben": "healthy storm"}
data_map = {"R11_control": "R11, no load", "R11_loaded": "R11, load"}
with (ROOT / "results/editorial/absolute_brier_ld.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        fields = [data_map.get(row["data"], row["data"]), name_map[row["mechanism"]],
                  row["count_gbdt"], row["presence_feature_gbdt"], row["unique_count_gbdt"]]
        latex_row = " & ".join(fields) + r" \\"
        assert latex_row in tex, latex_row

assert "DOI to be inserted" not in cover
assert "must be disclosed" not in cover
assert "rechecked immediately" not in cover
assert "To be completed" not in tex
assert "measure the theory's predictions" not in tex
assert "as Theorem~\\ref{thm:presence} predicts for any change" not in tex
print("submission asset checks passed; repository DOI must still be supplied before submission")
