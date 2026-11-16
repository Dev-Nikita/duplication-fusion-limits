"""Robustness check (declared 2026-10-24, after R11-X; NOT a replacement of the frozen X1-X3 values, which stay
primary): the frozen R11-X replication (replicate.py, unchanged) re-run with 10 seeds instead of 3.
Usage: python3 -I robust_seeds.py <units.csv> <out_dir> <full|alarm> <blocks>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "replicate.py")
src = open(path).read()
assert src.count("R.N_SEEDS = 3") == 1
sys.argv = [path] + sys.argv[1:]
exec(compile(src.replace("R.N_SEEDS = 3", "R.N_SEEDS = 10"), path, "exec"), {"__name__": "__main__", "__file__": path})
