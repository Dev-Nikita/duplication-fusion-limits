"""Restricted-Bayes frontier between a reference mechanism and the worst case (docs/frontier-protocol.md).
Analytical, post hoc. Usage:
  python3 -W ignore tools/frontier.py testbed
  python3 -I -W ignore tools/frontier.py flight <clean_flight dir> <flight_truth dir>
Outputs: results/frontier/frontier_{testbed,flight}.csv and a printed summary."""
import os, pickle, sys
import numpy as np, pandas as pd, cvxpy as cp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import finite_cap_cvx as C  # noqa: E402

EPS = [0, 0.05, 0.1, 0.2, 0.5, 1]; TIE = 1e-6


def restricted(q, p, D, eps):
    """q: 2 x (n_A) reference law of (y, A); p: 2 x (M+1) law of (y, U). Returns the rule x on a = 0..(D+1)M."""
    x, cons, c0, c1, M = C._ld_parts(p, D); n = (D + 1) * M + 1
    qq = np.zeros((2, n)); k = min(n, q.shape[1]); qq[:, :k] = q[:, :k]; qq[:, n - 1] += q[:, k:].sum(1)
    e = min(max(eps, TIE), 1 - TIE)
    ref = qq[0] @ cp.square(x) + qq[1] @ cp.square(1 - x)
    cp.Problem(cp.Minimize((1 - e) * ref + e * (p[0] @ c0 + p[1] @ c1)), cons).solve(solver=cp.CLARABEL)
    return np.clip(x.value, 0, 1)


def worst(x, p, D):
    I = C._intervals(p.shape[1] - 1, D)
    return float(sum(p[0, u] * np.max(x[I[u]] ** 2) + p[1, u] * np.max((1 - x[I[u]]) ** 2) for u in range(p.shape[1])))


def testbed():
    F = pickle.load(open("results/finite_cap/fold_laws.pkl", "rb")); out = []
    for f in F:
        ptr = f["n_train"] / f["n_train"].sum(); pte = f["n_test"] / f["n_test"].sum(); M = ptr.shape[1] - 1; u = np.arange(M + 1)
        for D in (1, 4, 16):
            for eps in EPS:
                x = restricted(ptr, ptr, D, eps)
                out.append(dict(dataset=f["dataset"], task=f["task"], held_block=f["held_block"], D=D, eps=eps,
                                ref_risk=float(pte[0] @ x[u] ** 2 + pte[1] @ (1 - x[u]) ** 2), worst_risk=worst(x, pte, D)))
    return pd.DataFrame(out), ["D", "eps"]


def flight(clean, truth):
    sys.path.insert(0, os.path.join(HERE, "..", "experiments", "flight"))
    import collections
    import flight_io as FIO, flight_study as ST, flight_audit as AU
    data = FIO.load(clean, truth); rows, items = AU.build(data, "B"); y = np.array([r["y"] for r in rows])
    days = sorted(items); week = {d: min(i // 7, 3) for i, d in enumerate(days)}; out = []
    for split, groups in (("week", [week[r["day"]] for r in rows]), ("airline", [r["airline"] for r in rows])):
        for g in sorted(set(groups)):
            te = np.array([v == g for v in groups]); tr = ~te; tr_days = {rows[i]["day"] for i in np.where(tr)[0]}
            if split == "airline":
                its = [it for d in items for fl, it in items[d].items() if fl.split("-")[0] != g]
            else:
                its = [it for d in tr_days for it in items[d].values()]
            cl = ST.clusters(its, AU.TAU); sizes = collections.Counter(cl.values()); D = max(sizes.values()) - 1; mc = len(sizes)
            Ftr = ST.features([rows[i] for i in np.where(tr)[0]], cl); Fte = ST.features([rows[i] for i in np.where(te)[0]], cl)
            n = (D + 1) * mc + 1; p = ST.law(y[tr], Ftr.U.values, mc); pte = ST.law(y[te], Fte.U.values, mc); q = ST.law(y[tr], Ftr.A.values, n - 1)
            ate = np.minimum(Fte.A.values.astype(int), n - 1)
            for eps in EPS:
                x = restricted(q, p, D, eps)
                out.append(dict(split=split, group=g, n=int(te.sum()), D_hat=D, n_clusters=mc, eps=eps,
                                train_outside_cap=float(np.mean(Ftr.A.values > (D + 1) * Ftr.U.values)),
                                ref_risk=float(np.mean((x[ate] - y[te]) ** 2)), worst_risk=worst(x, pte, D)))
    return pd.DataFrame(out), ["split", "eps"]


if __name__ == "__main__":
    os.makedirs("results/frontier", exist_ok=True)
    R, keys = testbed() if sys.argv[1] == "testbed" else flight(sys.argv[2], sys.argv[3])
    R.to_csv(f"results/frontier/frontier_{sys.argv[1]}.csv", index=False)
    w = R["n"] if "n" in R else pd.Series(1.0, index=R.index)
    S = R.assign(w=w).groupby(keys).apply(lambda d: pd.Series(dict(ref_risk=np.average(d.ref_risk, weights=d.w), worst_risk=np.average(d.worst_risk, weights=d.w)))).round(4)
    print(S.to_string())
    if "group" in R:
        print(R[R.split == "airline"].pivot(index="group", columns="eps", values="ref_risk").round(4).to_string())
        print(R[R.split == "airline"].pivot(index="group", columns="eps", values="worst_risk").round(4).to_string())
        print("share of training items outside the cap interval:", R.train_outside_cap.max())
