"""Word n-gram overlap between this manuscript and other texts (own prior manuscripts, close papers).

Usage: python -I tools/similarity_check.py paper/main.tex OTHER.txt [OTHER2.txt ...] [--n 8]
Reports, per other text, the share of the manuscript's n-grams that also occur there, and lists
the shared passages. Own manuscripts are NOT stored in this repository; pass their paths.
"""
import argparse
import re
import sys


def words(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    if path.endswith(".tex"):
        t = re.sub(r"%.*", " ", t)
        t = re.sub(r"\\(cite|ref|label|eqref)\{[^}]*\}", " ", t)
        t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
        t = re.sub(r"\\todo\{[^}]*\}", " ", t)
    t = re.sub(r"-\s*\n\s*", "", t)
    return re.findall(r"[a-z]+", t.lower())


def grams(w, n):
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manuscript")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    m = grams(words(a.manuscript), a.n)
    print(f"manuscript {a.n}-grams: {len(m)}")
    worst = 0.0
    for o in a.others:
        g = grams(words(o), a.n)
        shared = m & g
        share = len(shared) / max(1, len(m))
        worst = max(worst, share)
        print(f"{share*100:6.2f}%  {len(shared):5d} shared  {o}")
        for s in sorted(shared)[:15]:
            print("         ", s)
    sys.exit(0)


if __name__ == "__main__":
    main()
