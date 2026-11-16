# R4 — confirmation on the own Open5GS + UERANSIM testbed (2026-10-17)
Protocol: docs/r4-protocol.md (frozen before data existed; parser frozen on pilot only, sha256 ad9a7151…).
Campaign: 120 episodes (30 per class), 57 min, no interruptions, all NFs alive after every episode;
pilot (8 episodes) excluded. Raw: results/r4/features.csv, r4_cells.csv, r4_gate.csv.
Sources per episode: pcap (NGAP/NAS + SBI HTTP/2), AMF log, UE logs, Prometheus — four naturally
heterogeneous sources describing the same registrations.

## Learnability gate (calibration episodes, stacking AUROC ≥ 0.65)
| task | seeds passing | mean AUROC | status |
|---|---|---|---|
| amf_stall | 5/5 | 0.82 | included |
| reg_reject | 5/5 | 1.00 | included (trivially separable) |
| ausf_stall | 2/5 | 0.67 | **excluded** (fails in ≥ 3 seeds) |

## Confirmatory hypotheses (pooled over gated tasks × 5 seeds; paired bootstrap 95% CI)
| | estimate | CI | verdict |
|---|---|---|---|
| C1 stacking − A4 Brier at D = 8, h = 0.5 | 0.0147 | [0.0024, 0.0305] | **holds** |
| C2 A4 − stacking Brier at D = 1, h = 0.5 (< 0.01) | 0.0021 | [−0.0001, 0.0046] | **holds** |
| C3 A4 Brier change D = 0 → 8 with full lineage | 0.0000 | [0, 0] | **holds** |
| C4 certified-decision error − overall A4 error | −0.0055 | [−0.016, 0.005] | **not confirmed** |
**A4 is confirmed** by the pre-declared rule (C1 and C2 on gated tasks).

## Per task
* amf_stall: stacking Brier 0.201 (D=1) → 0.225 (D=8); A4 0.204 → 0.210 at h = 0.5, 0.202 flat at h = 0;
  C1 0.015 [0.002, 0.028]. Generative lineage model: AUROC 0.56 (again misspecified on real data).
* reg_reject: all methods separate perfectly (AUROC 1.0); stacking calibration degrades with D
  (0.0005 → 0.018), A4 stays ≤ 0.012.

## Honest limits
* Small test sets (≈ 18 episodes per task and seed); effects are smaller than on ORIGAMI (0.015 vs 0.038)
  — expected, as the stall signal is weaker.
* One of three fault families was not learnable at the injected intensities (ausf_stall); the pilot
  showed weak stalls are invisible. A larger campaign with stronger, pre-registered intensities is the
  next confirmation step.
* Duplication (D) and hidden lineage (h) are applied by the monitoring layer in analysis (controlled);
  the sources themselves are natural.

## Post-hoc (2026-10-18, exploratory, not in any protocol) — lineage-free normalisation baselines
`experiments/r4/posthoc_presence_baseline.py`, R4 data, 2 gated tasks × 5 random 60/40 splits, Brier:
| D, h | stacking (counts) | A4 | presence (counts > 0) | log1p(counts) |
|---|---|---|---|---|
| 1, 0.5 | 0.0750 | 0.0787 | 0.0796 | 0.0809 |
| 8, 0.5 | 0.0994 | 0.0876 | **0.0796** | 0.0889 |
| 8, 1.0 | 0.0994 | 0.0984 | **0.0796** | 0.0889 |
| 16, 1.0 | 0.1133 | 0.1092 | **0.0796** | 0.0927 |
**Consequence.** In this observation layer every slot has exactly one reading and every copy is derived, so
binarising counts removes duplication without any lineage and beats A4 whenever D shifts. A4 can only add
value where the same slot carries several *independent* units whose number is informative (e.g. many UEs
failing independently) mixed with derived re-emissions. The A4 claim as written is not supported against
this baseline; the observation layer must be redesigned (per-UE / per-procedure units) and re-tested.
