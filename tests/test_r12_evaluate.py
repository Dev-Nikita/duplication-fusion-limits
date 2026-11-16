import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "experiments" / "ext" / "rca_evaluate.py"


def test_empty_gate_returns_not_testable(tmp_path):
    rows = []
    for scheme in ("within", "cross"):
        for learner in ("gbdt", "logit"):
            for feature_set in ("COUNT", "UNIT", "BOTH", "RATE"):
                for system in ("S1", "S2", "S3"):
                    for case in ("c1", "c2"):
                        rows.append({
                            "scheme": scheme,
                            "test_system": system,
                            "learner": learner,
                            "set": feature_set,
                            "case": case,
                            "fault_dir": "f1",
                            "brier": 0.25,
                            "p_N": 0.5,
                            "p_F": 0.5,
                        })
    cells = tmp_path / "cells.csv"
    output = tmp_path / "results.json"
    pd.DataFrame(rows).to_csv(cells, index=False)

    completed = subprocess.run(
        [sys.executable, "-I", str(EVALUATOR), str(cells), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "NOT_TESTABLE"
    assert result["gate_systems"] == []
    assert result["decision"] == {
        "testable": False,
        "H_W": None,
        "H_WB": None,
        "H_A": None,
        "H_C": None,
        "GO": False,
    }
