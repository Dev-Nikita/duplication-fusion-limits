"""Block-level statistics and leave-one-physical-block-out sensitivity for Section 6.4 (analytical, existing data).
(1) Per held-out block: risk differences naive - minimax and presence - minimax (exact worst case over Delta_D on
    the held-out block's empirical law), averaged over the three tasks; R11 control/loaded share physical blocks.
(2) Sensitivity: for every physical block e, remove e from its campaign ENTIRELY (thresholds, training laws and
    held-out folds are recomputed without it) and recompute the fold-level summaries.
Rules: interior-point plug-in minimax (finite_cap_cvx), naive count rule, presence rule.
Usage: python3 -W ignore tools/block_sensitivity.py  -> results/finite_cap/block_level.csv, block_jackknife.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finite_cap_cvx as C, finite_cap_solvers as S  # noqa: E402
from apply_finite_cap import law, windows, TASKS, M  # noqa: E402

DS = (1, 4, 16)
SPECS = [("R10", "results/r10/units.csv", [13, 14, 15, 16, 17, 18], None),
         ("R6", "results/r6/units.csv", [4, 5, 6, 10, 11, 12], None),
         ("R11-control", "results/r11/units_cond.csv", list(range(21, 33)), "control"),
         ("R11-loaded", "results/r11/units_cond.csv", list(range(21, 33)), "loaded")]


def presence_rule(p, D):
    x = np.full((D + 1) * M + 1, p[1, 1:].sum() / max(p[:, 1:].sum(), 1e-300)); x[0] = p[1, 0] / max(p[:, 0].sum(), 1e-300)
    return x


def folds(u, blocks):
    out = []
    for task in TASKS:
        sub = u[u.label.isin(["healthy", task])].copy(); sub["y"] = (sub.label == task).astype(int)
        for b in blocks:
            tr, te = sub[sub.block != b], sub[sub.block == b]
            if te.y.nunique() < 2:
                continue
            thr = float(np.percentile(tr[tr.y == 0].ue_lat, 90))
            ptr, pte = law(windows(tr, thr)), law(windows(te, thr))
            eta = np.where(ptr.sum(0) > 0, ptr[1] / np.where(ptr.sum(0) > 0, ptr.sum(0), 1), np.nan)
            eta = pd.Series(eta).ffill().bfill().fillna(0.5).values
            for D in DS:
                x = C.ld_minimax(ptr, D)[1]; naive = eta[np.minimum(np.arange((D + 1) * M + 1), M)]
                out.append(dict(task=task, held_block=b, D=D, W_minimax=S.primal_value(x, pte, D),
                                W_naive=S.primal_value(naive, pte, D), W_presence=S.primal_value(presence_rule(ptr, D), pte, D)))
    return out


def summ(df):
    r = {}
    for D, z in df.groupby("D"):
        r[f"n_D{D}"] = len(z); r[f"mm_lt_naive_D{D}"] = (z.W_minimax < z.W_naive - 1e-12).mean()
        r[f"mm_lt_pres_D{D}"] = (z.W_minimax < z.W_presence - 1e-12).mean()
        r[f"mean_naive_minus_mm_D{D}"] = (z.W_naive - z.W_minimax).mean(); r[f"mean_pres_minus_mm_D{D}"] = (z.W_presence - z.W_minimax).mean()
    return r


data = {}
for name, path, blocks, cond in SPECS:
    u = pd.read_csv(path).replace([np.inf], 1e9); u = u[u.block.isin(blocks)]
    data[name] = (u[u.load_cond == cond] if cond else u, blocks)
full = []
for name, (u, blocks) in data.items():
    for r in folds(u, blocks):
        r["dataset"] = name; full.append(r)
full = pd.DataFrame(full); full.to_csv("results/finite_cap/block_full_rerun.csv", index=False)
bl = full.groupby(["dataset", "held_block", "D"]).agg(naive_minus_mm=("W_naive", "mean"), pres_minus_mm=("W_presence", "mean"),
                                                     mm=("W_minimax", "mean")).reset_index()
bl["naive_minus_mm"] -= bl.mm; bl["pres_minus_mm"] -= bl.mm; bl.to_csv("results/finite_cap/block_level.csv", index=False)
print("full:", summ(full), flush=True)
jk = []
phys = [("R10", b) for b in data["R10"][1]] + [("R6", b) for b in data["R6"][1]] + [("R11", b) for b in range(21, 33)]
for camp, e in phys:
    rows = []
    for name, (u, blocks) in data.items():
        refit = (name == camp) or (camp == "R11" and name.startswith("R11"))
        if refit:
            part = pd.DataFrame(folds(u[u.block != e], [b for b in blocks if b != e]))
        else:
            part = full[full.dataset == name]
        rows.append(part.assign(dataset=name))
    df = pd.concat(rows); r = summ(df); r.update(campaign=camp, excluded_block=e); jk.append(r); print(camp, e, flush=True)
pd.DataFrame(jk).to_csv("results/finite_cap/block_jackknife.csv", index=False)
