"""R11 frozen evaluation (docs/r11-protocol.md). Usage: python3 -I evaluate_r11.py <r11_cells.csv> <r11_gate.csv> <out.json>
Unit of inference = block (paired: both conditions in each block).
D_c(b)   = mean over primary tasks of max over the LD pair of [Brier(ref) - Brier(gbdt_unit)], in condition c
I(b)     = D_loaded(b) - D_control(b)                                         (H1)
I_LI(b)  = same contrast with the mean over the LI mechanisms instead of the LD worst case
H2(b)    = I(b) - I_LI(b)
ref = gbdt_mix_oracle (primary); gbdt_aug and gbdt_dedup (secondary, reported without a test).
Confirmed: one-sided 95 % t lower bound > 0 and >= 9/12 blocks positive (scaled: >= 3/4 of blocks).
Practically meaningful: estimate of I >= DELTA = 0.005 (the R10 negligibility margin)."""
import json, sys
import numpy as np, pandas as pd
from scipy import stats

DELTA, LDM = 0.005, ("storm_inc", "storm_ben")


def block_series(cells, ref):
    c = cells.copy(); c["d"] = c[ref] - c["gbdt_unit"]
    s = c.groupby(["cond", "task", "held_block", "mech"]).d.mean().reset_index()
    ld = s[s.mech.isin(LDM)].groupby(["cond", "task", "held_block"]).d.max().groupby(["cond", "held_block"]).mean().unstack(0)
    li = s[s.mech.str.startswith("li_")].groupby(["cond", "task", "held_block"]).d.mean().groupby(["cond", "held_block"]).mean().unstack(0)
    I = (ld["loaded"] - ld["control"]).dropna(); Ili = (li["loaded"] - li["control"]).dropna()
    return ld, I, Ili, (I - Ili).dropna()


def test(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    lb = m - stats.t.ppf(0.95, n - 1) * se
    return dict(est=round(m, 5), one_sided_lb95=round(lb, 5), t_ci95=[round(m - stats.t.ppf(.975, n - 1) * se, 5),
                round(m + stats.t.ppf(.975, n - 1) * se, 5)], positive=int((x > 0).sum()), n=n,
                confirmed=bool(lb > 0 and (x > 0).sum() >= int(np.ceil(0.75 * n))))


def main():
    cells, gate = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
    res = {"gate_auc_mean": gate.groupby(["cond", "task"]).auc_cal.mean().round(3).unstack(0).to_dict()}
    for ref in ("gbdt_mix_oracle", "gbdt_aug", "gbdt_dedup"):
        ld, I, Ili, H2 = block_series(cells, ref)
        r = {"D_control_mean": round(ld["control"].mean(), 5), "D_loaded_mean": round(ld["loaded"].mean(), 5),
             "H1_I": test(I), "LI_contrast": test(Ili), "H2_I_minus_ILI": test(H2)}
        for k in ("H1_I", "LI_contrast", "H2_I_minus_ILI"):
            r[k]["per_block"] = {int(b): round(float(v), 5) for b, v in {"H1_I": I, "LI_contrast": Ili, "H2_I_minus_ILI": H2}[k].items()}
        res[ref] = r
    p = res["gbdt_mix_oracle"]
    res["decision"] = dict(H1=p["H1_I"]["confirmed"], H2=p["H2_I_minus_ILI"]["confirmed"],
                           practically_meaningful=bool(p["H1_I"]["est"] >= DELTA))
    res["absolute_brier"] = cells.groupby(["cond", "mech"])[["augmented", "gbdt_aug", "gbdt_dedup", "gbdt_la", "gbdt_unit",
                                                             "gbdt_mix_oracle"]].mean().round(4).reset_index().to_dict("records")
    json.dump(res, open(sys.argv[3], "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("decision", "gate_auc_mean")}, indent=1))
    for ref in ("gbdt_mix_oracle", "gbdt_aug", "gbdt_dedup"):
        print(ref, {k: {kk: res[ref][k][kk] for kk in ("est", "one_sided_lb95", "positive", "n", "confirmed")}
                    for k in ("H1_I", "LI_contrast", "H2_I_minus_ILI")})


if __name__ == "__main__":
    main()
