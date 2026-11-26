"""Generates the LaTeX tables of the broker-redelivery section (main text and Supplementary S7) from a results
directory written by experiments/redelivery/evaluate.py and robust_risk.py. Used for the corrected analysis
(results/redelivery_corrected/A). Usage: python3 redelivery_tables.py <results dir> <out .tex>"""
import sys
import pandas as pd
RES, OUT = sys.argv[1], sys.argv[2]
B = pd.read_csv(f"{RES}/brier.csv"); AU = pd.read_csv(f"{RES}/audit.csv"); RR = pd.read_csv(f"{RES}/robust_risk.csv")
PB = pd.read_csv(f"{RES}/per_block.csv")
NAMES = [("presence", "Presence"), ("count_table", "Count, table"), ("count_gbdt", "Count, GBDT"), ("clipped_count", "Clipped count"),
         ("log_count", "Log count"), ("minimax", "Plug-in minimax ($D=2$)"), ("weighted", "Weighted ($\\varepsilon=0.2$)")]
f3 = lambda v: f"{v:.3f}"
def b(ds, reg, rule):
    return float(B[(B.dataset == ds) & (B.regime == reg) & (B.rule == rule)].brier.iloc[0])
nte = {ds: int(B[B.dataset == ds].n.iloc[0]) for ds in ("r10", "r6")}
out = {}
# main table: regimes
rows = []
for k, nm in NAMES + [("oracle_unique (reference)", "Unique count (reference)")]:
    if k.startswith("oracle"):
        rows.append("\\midrule")
    vals = [f3(b(ds, r, k)) for ds in ("r10", "r6") for r in ("S0", "S1", "S2", "S3a", "S4")]
    rows.append(nm + " & " + " & ".join(vals) + " \\\\")
out["tab:redelivery"] = ("\\begin{table}[t]\n\\centering\n\\caption{Broker redelivery: held-out Brier score by delivery regime (rules fitted under S1; R6 without the invalid blocks 7--9). The last row is a\nreference that needs lineage at test time.}\n\\label{tab:redelivery}\n\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{lcccccccccc}\n\\toprule\n"
    f" & \\multicolumn{{5}}{{c}}{{R10 ({nte['r10']} held-out episodes)}} & \\multicolumn{{5}}{{c}}{{R6 ({nte['r6']} held-out episodes)}} \\\\\n"
    "Rule & S0 & S1 & S2 & S3a & S4 & S0 & S1 & S2 & S3a & S4 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
# main robust table
rows = []
for k, nm in [NAMES[i] for i in (0, 1, 2, 3, 4, 6, 5)]:
    vals = []
    for ds in ("r10", "r6"):
        r = RR[(RR.dataset == ds) & (RR.rule == k)].iloc[0]
        vals += [f3(r.worst_observed), f3(r.count_only_D2), f3(r.label_aware_D2)]
    rows.append(nm + " & " + " & ".join(vals) + " \\\\")
out["tab:robust"] = ("\\begin{table}[t]\n\\centering\n\\caption{Broker redelivery, post hoc: worst observed held-out Brier score over S1, S2, S3a and S4, and exact model\nworst case on the held-out law for cap $D=2$ under a count-only and a label-aware mechanism. The model worst case is\nnot an observed risk.}\n\\label{tab:robust}\n\\scriptsize\n\\begin{tabular}{lcccccc}\n\\toprule\n"
    " & \\multicolumn{3}{c}{R10} & \\multicolumn{3}{c}{R6} \\\\\nRule & observed & count-only & label-aware & observed & count-only & label-aware \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
# supp audit
rows = []
for _, r in AU.iterrows():
    rows.append(f"{r.regime} & {r.deliveries:,} & {r.copies:,} & {r.share_roots_copied:.3f} & {r.max_multiplicity} \\\\")
n0 = int(AU[AU.regime == "S0"].deliveries.iloc[0])
out["tab:redaudit"] = ("\\begin{table}[h]\n\\centering\\scriptsize\n\\caption{Broker redelivery: audit of the trace without the invalid R6 blocks (" + f"{n0:,}" + " originals per regime). No delivery had an unknown original,\ntwo stream sequences, a gap in its delivery counter or more deliveries than configured, and every original was delivered.}\n\\label{tab:redaudit}\n\\begin{tabular}{lrrrr}\n\\toprule\nRegime & Deliveries & Copies & Share of reports copied & Largest multiplicity \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
rows = [nm.replace(" ($D=2$)", "").replace(" ($\\varepsilon=0.2$)", "") + f" & {f3(b('r10', 'S3b', k))} & {f3(b('r6', 'S3b', k))} \\\\" for k, nm in NAMES]
rows.append(f"Unique count (reference) & {f3(b('r10', 'S3b', 'oracle_unique (reference)'))} & {f3(b('r6', 'S3b', 'oracle_unique (reference)'))} \\\\")
out["tab:redfull"] = ("\\begin{table}[h]\n\\centering\\scriptsize\n\\caption{Broker redelivery: held-out Brier score in regime S3b (redelivery limit above the declared cap $D=2$;\noutside the model).}\n\\label{tab:redfull}\n\\begin{tabular}{lcc}\n\\toprule\nRule & R10 & R6 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
rows = []
for k, nm in [NAMES[i] for i in (0, 1, 2, 3, 4, 6, 5)]:
    vals = []
    for ds in ("r10", "r6"):
        r = RR[(RR.dataset == ds) & (RR.rule == k)].iloc[0]
        vals += [f"{r[f'label_aware_D{d}']:.3f}"[1:] + "/" + f"{r[f'count_only_D{d}']:.3f}"[1:] for d in range(4)]
    rows.append(nm.replace(" ($D=2$)", "").replace(" ($\\varepsilon=0.2$)", "") + " & " + " & ".join(vals) + " \\\\")
out["tab:redcap"] = ("\\begin{table}[h]\n\\centering\\scriptsize\n\\caption{Broker redelivery, post hoc: exact model worst case on the held-out law by cap $D$ (label-aware / count-only\nmechanism). Rules fitted for $D=2$ under S1 and not refitted. Counts above the range on which a rule was fitted\n(above $3M$ for the count table, the minimax and the weighted rule) take the last value of the rule. The optimality\nof the minimax rule holds for $D=2$ and the training law only; the columns for other caps are a descriptive stress test.}\n\\label{tab:redcap}\n\\tiny\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lcccccccc}\n\\toprule\n"
    " & \\multicolumn{4}{c}{R10} & \\multicolumn{4}{c}{R6} \\\\\nRule & $D=0$ & $D=1$ & $D=2$ & $D=3$ & $D=0$ & $D=1$ & $D=2$ & $D=3$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
w = PB[PB.regime.isin(["S1", "S2", "S3a", "S4"])].groupby(["dataset", "rule", "block"]).brier.max()
blocks = [("r6", bl) for bl in sorted(PB[PB.dataset == "r6"].block.unique())] + [("r10", bl) for bl in sorted(PB[PB.dataset == "r10"].block.unique())]
rows = []
for k, nm in [NAMES[i] for i in (0, 1, 2, 3, 4, 6, 5)]:
    rows.append(nm.replace(" ($D=2$)", "").replace(" ($\\varepsilon=0.2$)", "") + " & " + " & ".join(f3(w[(ds, k, bl)]) for ds, bl in blocks) + " \\\\")
out["tab:redblock"] = ("\\begin{table}[h]\n\\centering\\scriptsize\n\\caption{Broker redelivery: worst observed Brier score over S1, S2, S3a and S4 by held-out physical block (50\nepisodes each). Five blocks; descriptive.}\n\\label{tab:redblock}\n\\begin{tabular}{lccccc}\n\\toprule\n"
    " & \\multicolumn{3}{c}{R6} & \\multicolumn{2}{c}{R10} \\\\\nRule & " + " & ".join(f"block {bl}" for _, bl in blocks) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
with open(OUT, "w") as f:
    for k, v in out.items():
        f.write(f"%%% {k}\n{v}\n")
print("tables:", list(out))
