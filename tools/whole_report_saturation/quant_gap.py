"""R7: quantitative whole-report saturation. For every law of the 315-law study (same seed and order) and every cap
D = 0..D_sat, computes exactly (rationals)
  G_D   = R*_inf - R*_D                                   (saturation gap)
  UB    = sum_S h_S delta_{D,S}                            (candidate upper bound)
  LBq   = sum_S h_S kappa_S delta_{D,S}^2                  (Prop.-5-type quadratic lower bound, tested, not claimed)
with S the mixed blocks of the unrestricted optimum, h_S = pi(S) theta_S (1-theta_S), kappa_S = 4 theta_S(1-theta_S),
delta_{D,S} = 1 - max partial coupling of p(.|0,S) and p(.|1,S) on D-compatible pairs.
Usage: python quant_gap.py CONFIG_INDEX"""
import sys, csv, random
from fractions import Fraction as F
from finite_cap import Game, gen_law_iid, gen_law_sparse
from exact_core import build_atoms, below_from_edges, isotonic, value, Flow
from cert_audit import CONFIGS

def blocks_inf(law, g):
    atoms, st = build_atoms(law)
    val, _ = isotonic(atoms, below_from_edges(atoms, st, g.inf_edge))
    lev = {}
    for a, v in val.items():
        lev.setdefault(v, []).append(a)
    return atoms, st, val, {v: S for v, S in lev.items() if 0 < v < 1}

def delta(atoms, st, S, edge):
    z = [i for i in S if atoms[i][0] == 0]; o = [i for i in S if atoms[i][0] == 1]
    p0 = sum(atoms[i][1] for i in z); p1 = sum(atoms[i][1] for i in o)
    idx = {a: k + 2 for k, a in enumerate(S)}; fl = Flow(len(S) + 2)
    for i in z:
        fl.add(0, idx[i], atoms[i][1] / p0)
    for j in o:
        fl.add(idx[j], 1, atoms[j][1] / p1)
    for i in z:
        for j in o:
            if edge(st[i], st[j]):
                fl.add(idx[i], idx[j], F(2))
    return 1 - fl.maxflow(0, 1), p0, p1

def main(ci):
    rnd = random.Random(1117); rows = []
    for k, (K, M, kind, ntr) in enumerate(CONFIGS):
        g = Game(K, M, 3 * M + 2) if k == ci else None
        for tr in range(ntr):
            law = gen_law_iid(K, M, rnd) if kind == "iid" else gen_law_sparse(K, M, rnd, rnd.randint(4, 14))
            if k != ci:
                continue
            atoms, st, vinf, B = blocks_inf(law, g); Rinf = value(atoms, vinf)
            for D in range(0, 3 * M + 3):
                a2, s2 = build_atoms(law); vD, _ = isotonic(a2, below_from_edges(a2, s2, g.edge_D(D))); RD = value(a2, vD)
                G = Rinf - RD; UB = F(0); LB = F(0)
                for th, S in B.items():
                    d, p0, p1 = delta(atoms, st, S, g.edge_D(D)); h = (p0 * p1) / (p0 + p1); kap = 4 * th * (1 - th)
                    UB += h * d; LB += h * kap * d * d
                rows.append(dict(K=K, M=M, kind=kind, trial=tr, D=D, G=str(G), UB=str(UB), LBq=str(LB), G_f=float(G), UB_f=float(UB),
                                 LBq_f=float(LB), ub_ok=G <= UB, lbq_ok=LB <= G, tight=(G == UB), blocks=len(B)))
                if G == 0:
                    break
    fn = f"out/quant_gap_cfg{ci}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"cfg {ci} {CONFIGS[ci]}: rows {len(rows)}; UB violations {sum(not r['ub_ok'] for r in rows)}; "
          f"quadratic-LB violations {sum(not r['lbq_ok'] for r in rows)}; tight {sum(r['tight'] and r['G_f']>0 for r in rows)}; "
          f"positive-gap rows {sum(r['G_f']>0 for r in rows)}")

if __name__ == "__main__":
    main(int(sys.argv[1]))
