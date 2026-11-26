import os, subprocess, sys
import pytest
pytest.importorskip("cvxpy")


def test_whole_report_marginal_checks():
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    out = subprocess.run([sys.executable, "-W", "ignore", os.path.join(root, "tools", "whole_report_marginal.py")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert " 0 failures" in out.stdout, out.stdout
