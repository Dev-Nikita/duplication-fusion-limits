"""Evaluation for the Flight study (protocol docs/flight-protocol.md). Brier per rule and split; paired contrasts with
a day-cluster bootstrap (2,000 resamples, seed fixed). Usage: python3 -I flight_evaluate.py <out_dir>"""
import json, os, sys
import numpy as np, pandas as pd

out = sys.argv[1]; P = pd.read_csv(os.path.join(out, "flight_predictions.csv")); rng = np.random.default_rng(20261115)
RULES = ["count", "unique", "presence", "count_neg", "unique_neg", "naive", "minimax"]
for r in RULES:
    P["b_" + r] = (P["p_" + r] - P.y) ** 2
CONTRASTS = [("H-F1/H-F2", "count", "unique"), ("H-F3", "count_neg", "unique"), ("aux", "presence", "unique"), ("aux", "count", "presence"),
             ("aux", "naive", "unique"), ("aux", "minimax", "naive"), ("aux", "minimax", "presence"), ("aux", "unique_neg", "unique")]
rows = []
for (tau, split), z in P.groupby(["tau", "split"]):
    days = sorted(z.day.unique()); byday = {d: z[z.day == d] for d in days}
    r = dict(tau=tau, split=split, n=len(z), n_days=len(days), D_hat=f"{z.D_hat.min()}-{z.D_hat.max()}", n_clusters=f"{z.n_clusters.min()}-{z.n_clusters.max()}")
    for k in RULES:
        r["brier_" + k] = z["b_" + k].mean()
    boot = {c: [] for c in CONTRASTS}
    for _ in range(2000):
        s = pd.concat([byday[d] for d in rng.choice(days, size=len(days), replace=True)])
        for c in CONTRASTS:
            boot[c].append(s["b_" + c[1]].mean() - s["b_" + c[2]].mean())
    for c in CONTRASTS:
        d = z["b_" + c[1]].mean() - z["b_" + c[2]].mean(); q = np.quantile(boot[c], [0.05, 0.95])
        r[f"d_{c[1]}_minus_{c[2]}"] = d; r[f"lo90_{c[1]}_minus_{c[2]}"] = q[0]; r[f"hi90_{c[1]}_minus_{c[2]}"] = q[1]
    # worst group
    g = z.groupby("group")[["b_count", "b_unique", "b_presence", "b_count_neg"]].mean()
    r["groups"] = json.dumps({str(k): {c: round(float(v), 4) for c, v in row.items()} for k, row in g.iterrows()})
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "flight_results.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(R[["tau", "split", "n", "D_hat", "n_clusters"] + ["brier_" + k for k in RULES]].round(4).to_string(index=False))
print(R[["tau", "split"] + [c for c in R.columns if c.startswith(("d_", "lo90_", "hi90_"))]].round(4).T.to_string())
