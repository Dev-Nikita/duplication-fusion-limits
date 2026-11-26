"""Regression test: the data-integrity gate must pass (invalid R6 blocks 7-9 unused outside raw and withdrawn files,
corrected replay clean, Flight gold sites removed, stated episode counts consistent, withdrawn analysis marked)."""
import os, subprocess, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def test_data_integrity_gate():
    r = subprocess.run([sys.executable, "tools/data_integrity_audit.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT: PASS" in r.stdout
