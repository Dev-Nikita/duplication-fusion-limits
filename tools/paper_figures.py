"""Figures of the article and the supplementary material, drawn only from files in results/ (and, for the worked example
of Figure 2(a), from an exact computation on a small law defined below). Writes PDF files to results/figures/.
Usage (from the repository root): python3 -W ignore tools/paper_figures.py"""
import glob, json, math, os, sys
from fractions import Fraction as F
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_blind_saturation import dsat_aware, dsat_blind  # noqa: E402

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)
# Okabe-Ito palette (colour-blind safe); every series also has its own marker or hatch for grey-scale printing
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7", sky="#56B4E9",
         yellow="#F0E442", black="#000000", grey="#7f7f7f")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "lines.linewidth": 1.3, "axes.linewidth": 0.7,
                     "pdf.fonttype": 42, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("written", f"{OUT}/{name}.pdf")


# ------------------------------------------------------------------ Figure 2: saturation and mechanism information
def value_one_type(p, D, label_aware):
    """exact finite-cap minimax value for one report type (interior point); p: 2 x (M+1) masses."""
    import cvxpy as cp
    M = p.shape[1] - 1; x = cp.Variable((D + 1) * M + 1); terms = []
    for u in range(M + 1):
        I = [0] if u == 0 else list(range(u, (D + 1) * u + 1))
        if p[:, u].sum() == 0:
            continue
        if label_aware:
            terms.append(p[0, u] * cp.max(cp.square(x[I])) + p[1, u] * cp.max(cp.square(1 - x[I])))
        else:
            terms.append(cp.max(cp.hstack([p[0, u] * cp.square(x[a]) + p[1, u] * cp.square(1 - x[a]) for a in I])))
    pr = cp.Problem(cp.Minimize(cp.sum(cp.hstack(terms))), [x >= 0, x <= 1]); pr.solve(solver="CLARABEL")
    return pr.value


def bayes(p):
    return float(sum(a * b / (a + b) for a, b in zip(p[0], p[1]) if a + b > 0))


def fig_saturation():
    # worked example (M = 8): class laws of the number of firing sources
    M = 8
    p = np.zeros((2, M + 1))
    p[0, [0, 1, 2, 3]] = [0.20, 0.15, 0.10, 0.05]
    p[1, [0, 2, 4, 8]] = [0.05, 0.15, 0.15, 0.15]
    mu = [{u: F(round(p[y, u] * 100), round(p[y, 1:].sum() * 100)) for u in range(1, M + 1) if p[y, u] > 0} for y in (0, 1)]
    da, du = dsat_aware(*mu), dsat_blind(*mu)
    Ds = list(range(0, M))
    va = [value_one_type(p, D, True) for D in Ds]; vu = [value_one_type(p, D, False) for D in Ds]
    pres = np.array([[p[0, 0], p[0, 1:].sum()], [p[1, 0], p[1, 1:].sum()]])
    rbu, rbp = bayes(p), bayes(pres)
    f = pd.read_csv("results/label_blind/folds.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    a = ax[0]
    a.axhline(rbp, color=C["grey"], ls="--", lw=1); a.text(0.15, rbp + 0.002, r"$R_{\mathrm{B}}(P)$, presence", ha="left", va="bottom", fontsize=7)
    a.axhline(rbu, color=C["grey"], ls=":", lw=1); a.text(0.25, rbu + 0.002, r"$R_{\mathrm{B}}(U)$, unique counts", ha="left", va="bottom", fontsize=7)
    a.plot(Ds, va, "-o", ms=3.5, color=C["orange"], label="label-aware $\\Delta_D$")
    a.plot(Ds, vu, "--s", ms=3.5, color=C["blue"], label="count-only $\\Delta_D^U$")
    for d, col, lab in ((da, C["orange"], r"$D_{\mathrm{sat}}$"), (du, C["blue"], r"$D_{\mathrm{sat}}^U$")):
        a.axvline(d, color=col, lw=0.8, ls="-.", alpha=0.8); a.text(d - 0.1, rbu + 0.012, lab, color=col, fontsize=7.5, ha="right")
    a.set_xlabel("copy cap $D$"); a.set_ylabel("minimax Brier risk $R^*$"); a.set_xticks(Ds)
    a.set_ylim(rbu - 0.01, rbp + 0.022)
    a.legend(loc="center left", frameon=False, bbox_to_anchor=(0.30, 0.45)); a.set_title("(a) worked example, $M=8$", loc="left")
    b = ax[1]
    rng = np.random.default_rng(0); j = lambda n: rng.uniform(-0.25, 0.25, n)
    b.scatter(f.D_sat_aware + j(len(f)), f.D_sat_countonly + j(len(f)), s=9, color=C["blue"], alpha=0.7, lw=0)
    m = max(f.D_sat_aware.max(), f.D_sat_countonly.max()) + 1
    b.plot([0, m], [0, m], color=C["black"], lw=0.8); b.set_xlim(-0.5, m); b.set_ylim(-0.5, m)
    b.set_xlabel(r"$D_{\mathrm{sat}}$ (label-aware)"); b.set_ylabel(r"$D_{\mathrm{sat}}^U$ (count-only)")
    b.set_title("(b) 108 testbed training laws", loc="left")
    c = ax[2]
    diff = f.D_sat_countonly - f.D_sat_aware
    c.hist(diff, bins=np.arange(-0.5, diff.max() + 1.5, 1), color=C["blue"], edgecolor="white", lw=0.4)
    c.set_xlabel(r"$D_{\mathrm{sat}}^U-D_{\mathrm{sat}}$"); c.set_ylabel("folds")
    c.set_title(f"(c) equal in {int((diff == 0).sum())}, larger in {int((diff > 0).sum())}", loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_saturation")
    json.dump(dict(M=M, D_sat=da, D_sat_U=du, RB_U=rbu, RB_P=rbp, label_aware=va, count_only=vu),
              open(f"{OUT}/fig_saturation_values.json", "w"), indent=1)


# ------------------------------------------------------------------ Figure 4: Proposition 8 gap and bound
def fig_gap_bound():
    q = pd.concat([pd.read_csv(x) for x in sorted(glob.glob("results/whole_report/out/quant_gap_cfg*.csv"))])
    q = q[q.G_f > 0]
    t = pd.read_csv("results/whole_report/out/tolerance_caps.csv").sort_values("eps")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = ax[0]
    tight = q.tight.astype(str).str.lower() == "true"
    a.scatter(q.UB_f[~tight], q.G_f[~tight], s=8, color=C["blue"], alpha=0.6, lw=0, label=f"bound not attained ({int((~tight).sum())})")
    a.scatter(q.UB_f[tight], q.G_f[tight], s=12, marker="^", color=C["red"], alpha=0.8, lw=0, label=f"attained ({int(tight.sum())})")
    m = q.UB_f.max() * 1.05; a.plot([0, m], [0, m], color=C["black"], lw=0.8)
    a.set_xlim(0, m); a.set_ylim(0, m)
    a.set_xlabel(r"bound $B_D=\sum_S h_S\,\delta_{D,S}$"); a.set_ylabel(r"exact gap $G_D=R^*_\infty-R^*_D$")
    a.legend(frameon=False, loc="upper left"); a.set_title(f"(a) {len(q)} law–cap cases with $G_D>0$", loc="left")
    b = ax[1]
    x = np.arange(len(t)); n = t.laws.values
    parts = [(t.equal / n, "exact = certified", C["green"], ""), (t.larger_by_1 / n, "certified larger by 1", C["sky"], "//"),
             (t.larger_by_2plus / n, "certified larger by 2–3", C["red"], "xx")]
    bottom = np.zeros(len(t))
    for v, lab, col, h in parts:
        b.bar(x, v, bottom=bottom, color=col, hatch=h, edgecolor="white", lw=0.4, label=lab, width=0.65); bottom += v
    for i, v in enumerate(t.equal / n):
        b.text(i, v / 2, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7)
    b.set_xticks(x); b.set_xticklabels([f"{e:g}" for e in t.eps]); b.set_xlabel(r"tolerance $\varepsilon$")
    b.set_ylabel("share of 315 laws"); b.set_ylim(0, 1); b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.5, columnspacing=0.8, handlelength=1.2)
    b.set_title(r"(b) tolerance caps $D_\varepsilon^{\mathrm{exact}}$ vs $D_\varepsilon^{\mathrm{cert}}$", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_gap_bound")


# ------------------------------------------------------------------ Figure 5: plug-in minimax risk versus copy cap
def fig_copycap():
    s = pd.read_csv("results/finite_cap/summary.csv")
    ds = list(dict.fromkeys(s.dataset)); tasks = ["amf_stall", "ausf_stall", "net_delay"]
    sty = dict(zip(ds, [(C["blue"], "o", "-"), (C["orange"], "s", "--"), (C["green"], "^", "-."), (C["purple"], "D", ":")]))
    names = {"R10": "R10", "R6": "R6", "R11-control": "R11, no load", "R11-loaded": "R11, load"}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, t in zip(axs, tasks):
        for k in ds:
            z = s[(s.dataset == k) & (s.task == t)].sort_values("D")
            if z.empty:
                continue
            col, mk, ls = sty[k]; xs = np.arange(len(z))
            ax.plot(xs, z.R_star, ls=ls, marker=mk, ms=3.5, color=col, label=names.get(k, k))
        xs = np.arange(len(z)); ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in z.D])
        ax.set_title(t.replace("_", " ").replace("amf", "AMF").replace("ausf", "AUSF").replace("net delay", "network delay"), loc="left")
        ax.set_xlabel("copy cap $D$"); ax.grid(alpha=0.25, lw=0.5)
    axs[0].set_ylabel(r"minimax risk $R^*$ (training law)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_copycap")


# ------------------------------------------------------------------ Figure 6: broker redelivery
RULES = [("presence", "presence"), ("count_table", "count table"), ("count_gbdt", "count GBDT"),
         ("clipped_count", "clipped counts"), ("log_count", "log counts"), ("minimax", "plug-in minimax"),
         ("weighted", "weighted")]


def fig_redelivery():
    r = pd.read_csv("results/redelivery_corrected/B/robust_risk.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.55), sharey=True)
    for ax, (ds, title) in zip(axs, (("r10", "(a) campaign R10"), ("r6", "(b) campaign R6"))):
        z = r[r.dataset == ds].set_index("rule")
        x = np.arange(len(RULES)); w = 0.26
        for i, (col, lab, c, h) in enumerate((("worst_observed", "worst observed regime", C["grey"], ""),
                                               ("count_only_D2", "count-only worst case, $D=2$", C["sky"], "//"),
                                               ("label_aware_D2", "label-aware worst case, $D=2$", C["blue"], ""))):
            ax.bar(x + (i - 1) * w, [z.loc[k, col] for k, _ in RULES], w, color=c, hatch=h, edgecolor="white", lw=0.4, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in RULES], rotation=35, ha="right"); ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axs[0].set_ylabel("Brier risk"); axs[0].set_ylim(0, 0.2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_redelivery")


# ------------------------------------------------------------------ Supplementary figures
def figS_redelivery_cap():
    r = pd.read_csv("results/redelivery_corrected/B/robust_risk.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    marks = ["o", "s", "^", "v", "D", "P", "X"]
    for ax, (ds, title) in zip(axs, (("r10", "(a) R10"), ("r6", "(b) R6"))):
        z = r[r.dataset == ds].set_index("rule")
        for (k, lab), mk, col in zip(RULES, marks, [C["grey"], C["orange"], C["purple"], C["sky"], C["green"], C["blue"], C["red"]]):
            ax.plot(range(4), [z.loc[k, f"label_aware_D{d}"] for d in range(4)], marker=mk, ms=3.5, color=col, label=lab)
        ax.set_xticks(range(4)); ax.set_xlabel("copy cap $D$"); ax.set_title(title, loc="left"); ax.grid(alpha=0.25, lw=0.5)
    axs[0].set_ylabel("label-aware worst-case Brier risk")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(w_pad=0.8)
    save(fig, "figS_redelivery_cap")


def figS_frontier():
    t = pd.read_csv("results/frontier/frontier_testbed.csv").groupby(["D", "eps"])[["ref_risk", "worst_risk"]].mean().reset_index()
    f = pd.read_csv("results/frontier/frontier_flight.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    a = axs[0]
    for (D, z), mk, col in zip(t.groupby("D"), ["o", "s", "^"], [C["blue"], C["orange"], C["green"]]):
        a.plot(z.ref_risk, z.worst_risk, marker=mk, ms=3.5, color=col, label=f"$D={D}$")
        for _, row in z.iterrows():
            if row.eps in (0.0, 1.0):
                a.annotate(f"$\\varepsilon={row.eps:g}$", (row.ref_risk, row.worst_risk), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    a.set_xlabel("held-out risk under the training mechanism"); a.set_ylabel("held-out worst-case risk")
    a.legend(frameon=False); a.set_title("(a) testbed, mean over 108 folds", loc="left")
    b = axs[1]
    cols = [c for c in f.columns]
    keyD = "D" if "D" in cols else None
    grp = [c for c in ("split", "D") if c in cols]
    for (key, z), mk, col in zip(f.groupby(grp), ["o", "s", "^", "v", "D", "P"], [C["blue"], C["orange"], C["green"], C["purple"], C["red"], C["sky"]]):
        z = z.groupby("eps")[["ref_risk", "worst_risk"]].mean().reset_index()
        lab = {"airline": "held-out airline", "week": "held-out date block"}.get(key if not isinstance(key, tuple) else key[0], str(key))
        b.plot(z.ref_risk, z.worst_risk, marker=mk, ms=3.5, color=col, label=lab)
        for _, row in z.iterrows():
            if row.eps in (0.0, 0.2):
                b.annotate(f"$\\varepsilon={row.eps:g}$", (row.ref_risk, row.worst_risk), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    b.set_xlabel("held-out risk under the natural copying"); b.set_ylabel("held-out worst-case risk")
    b.legend(frameon=False, fontsize=6.5); b.set_title("(b) Flight", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "figS_frontier")


def figS_flight():
    d = pd.read_csv("results/flight/audit_R2/flight_gold_exclusion_results.csv")
    d = d[d["mode"] == "A"].set_index("split")
    rules = [("brier_majority", r"hard majority vote $\mathbf{1}\{A>N\}$", "neg"), ("brier_coverage", "coverage only ($A+N$)", "neg"),
             ("brier_presence", "presence $P$", "count"), ("brier_minimax", r"plug-in minimax, cap $\hat D$", "count"),
             ("brier_unique", "unique clusters $U$", "lineage"), ("brier_count", "count $A$", "count"),
             ("brier_unique_neg", "unique clusters with negatives", "lineage"), ("brier_neg_only", "negatives only $N$", "neg"),
             ("brier_count_neg", "count with negatives", "neg"), ("brier_frac", "firing fraction $A/(A+N)$", "neg")]
    col = {"count": C["blue"], "neg": C["orange"], "lineage": C["green"]}
    hat = {"count": "", "neg": "//", "lineage": ".."}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (sp, title) in zip(axs, (("week", "(a) held-out date block"), ("airline", "(b) held-out airline"))):
        y = np.arange(len(rules))[::-1]
        ax.barh(y, [d.loc[sp, k] for k, _, _ in rules], color=[col[g] for _, _, g in rules],
                hatch=None, edgecolor="white", lw=0.4)
        for yi, (k, _, g) in zip(y, rules):
            if hat[g]:
                ax.barh(yi, d.loc[sp, k], color="none", hatch=hat[g], edgecolor="white", lw=0)
        ax.set_yticks(y); ax.set_yticklabels([l for _, l, _ in rules]); ax.set_xlabel("Brier score"); ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=col[g], hatch=hat[g], edgecolor="white") for g in ("count", "neg", "lineage")],
               ["positive reports only", "uses negative reports", "uses a lineage proxy (clusters)"],
               frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.06))
    fig.tight_layout(w_pad=1.0)
    save(fig, "figS_flight")


if __name__ == "__main__":
    fig_saturation(); fig_gap_bound(); fig_copycap(); fig_redelivery()
    figS_redelivery_cap(); figS_frontier(); figS_flight()
