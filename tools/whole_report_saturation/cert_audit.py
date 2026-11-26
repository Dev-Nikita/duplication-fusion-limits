"""P0.5: dump exact KKT certificates for the 315 laws of `finite_cap.py laws` (same seed, same order)
and run an independent numerical test of Theorem B (CLARABEL values + SciPy transport LP).
Usage: python cert_audit.py CONFIG_INDEX   (0..7). Writes out/certs/cfg{i}.jsonl and out/certs/indep{i}.csv"""
import sys, json, csv, random
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
from finite_cap import Game, gen_law_iid, gen_law_sparse, generic_qp
from exact_core import build_atoms, below_from_edges, isotonic, certify, value, kkt_multipliers

CONFIGS = [(2, 3, "iid", 40), (2, 4, "iid", 30), (2, 3, "sparse", 60), (2, 4, "sparse", 40),
           (3, 2, "iid", 30), (3, 2, "sparse", 60), (3, 3, "sparse", 40), (3, 3, "iid", 15)]

def exact_solve(law, edge):
    atoms, st = build_atoms(law)
    below = below_from_edges(atoms, st, edge)
    val, _ = isotonic(atoms, below)
    assert certify(atoms, below, val)
    mu = kkt_multipliers(atoms, below, val)
    return atoms, st, val, mu, value(atoms, val)

def transport_ok(atoms, st, Sidx, theta, edge):
    """Independent float LP: coupling inside block Sidx on edges `edge`."""
    zero = [i for i in Sidx if atoms[i][0] == 0]; one = [i for i in Sidx if atoms[i][0] == 1]
    pairs = [(n, m) for n in zero for m in one if edge(st[n], st[m])]
    if not pairs:
        return False
    A = np.zeros((len(zero) + len(one), len(pairs))); b = []
    for j, (n, m) in enumerate(pairs):
        A[zero.index(n), j] = 1; A[len(zero) + one.index(m), j] = 1
    b = [float(atoms[n][1] * theta) for n in zero] + [float(atoms[m][1] * (1 - theta)) for m in one]
    sc = max(b)
    r = linprog(np.zeros(len(pairs)), A_eq=A, b_eq=np.array(b) / sc, bounds=[(0, None)] * len(pairs), method="highs")
    return r.status == 0

def main(ci):
    rnd = random.Random(1117)
    out_c = open(f"out/certs/cfg{ci}.jsonl", "w")
    rows = []
    for k, (K, M, kind, ntr) in enumerate(CONFIGS):
        g = Game(K, M, 3 * M + 2) if k == ci else None
        for tr in range(ntr):
            law = gen_law_iid(K, M, rnd) if kind == "iid" else gen_law_sparse(K, M, rnd, rnd.randint(4, 14))
            if k != ci:
                continue
            atoms, st, vinf, muinf, Vinf = exact_solve(law, g.inf_edge)
            levels = {}
            for a, v in vinf.items():
                levels.setdefault(v, []).append(a)
            blocks = [(v, S) for v, S in levels.items() if 0 < v < 1]
            Vinf_cl = generic_qp(g, law, 3 * M + 2, "CLARABEL")
            dsat = None
            D = 0
            while True:
                a2, s2, vD, muD, VD = exact_solve(law, g.edge_D(D))
                out_c.write(json.dumps(dict(K=K, M=M, kind=kind, trial=tr, D=D,
                    atoms=[[y, list(n), str(w)] for (y, w), n in zip(a2, s2)],
                    val={str(i): str(v) for i, v in vD.items()},
                    mu=[[n, m, str(x)] for (n, m), x in muD.items()], value=str(VD))) + "\n")
                exact_eq = (VD == Vinf)
                cl = generic_qp(g, law, D, "CLARABEL")
                lp_crit = all(transport_ok(atoms, st, S, v, g.edge_D(D)) for v, S in blocks)
                rows.append(dict(K=K, M=M, kind=kind, trial=tr, D=D, exact_equal=exact_eq,
                                 lp_criterion=lp_crit, clarabel_gap=Vinf_cl - cl,
                                 clarabel_equal_1e7=abs(Vinf_cl - cl) < 1e-7))
                if exact_eq and dsat is None:
                    dsat = D
                if dsat is not None and D >= dsat + 1 or D >= 3 * M + 2:
                    break
                D += 1
            out_c.write(json.dumps(dict(K=K, M=M, kind=kind, trial=tr, D="inf",
                atoms=[[y, list(n), str(w)] for (y, w), n in zip(atoms, st)],
                val={str(i): str(v) for i, v in vinf.items()},
                mu=[[n, m, str(x)] for (n, m), x in muinf.items()], value=str(Vinf))) + "\n")
    with open(f"out/certs/indep{ci}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if not (r["exact_equal"] == r["lp_criterion"] == r["clarabel_equal_1e7"])]
    print(f"cfg {ci} {CONFIGS[ci]}: {len(rows)} (law,D) rows; disagreements exact/LP/CLARABEL: {len(bad)}; "
          f"min nonzero clarabel gap among exact-unequal: {min([r['clarabel_gap'] for r in rows if not r['exact_equal']], default=None)}")
    for r in bad[:5]: print("  ", r)

if __name__ == "__main__":
    main(int(sys.argv[1]))
