# R7 feasibility pilot — natural repeated events (2026-10-19)
Question (decision rule agreed before looking): do the existing testbed logs contain natural repeated
events, frequent enough and classifiable into (a) exact information copies, (b) retries that carry new
state, (c) independent observations, to support a natural-duplication R7? If not, no R7 is run and the
paper reports the limitation. Script: experiments/r7/pilot_natural_repeats.py (counts only; no outcome model).

## Findings over all 1,028 recorded episodes (R4 pilot/campaign, R5, R6 incl. excluded blocks)
* Registration attempts per UE: exactly one in every episode and every class (0 UEs with a retry).
* NAS timer expiries (T3510) / retransmissions: in 1 of 750 analysed campaign episodes (ep_0432, AMF stall,
  12 UEs); 0 elsewhere.
* The only systematic per-UE repetition is the second Authentication Request after SQN
  resynchronisation — a deterministic protocol step present in every class (2.00 per UE), i.e. not a copy
  and not informative.
* AMF 'epoll failed' lines (2–10 per episode) are produced by the SIGSTOP fault injection itself; using
  them would measure the injector, not the monitoring pipeline.
* Repeated AMF log templates (≈ 200–330 per episode) are per-UE lines of the same template, i.e.
  different events, not copies.
* Alertmanager (considered alternative): Open5GS Prometheus metrics are aggregate, not per UE, and default
  grouping/repeat intervals exceed the 20 s episode, so it would yield essentially no repeats.

## Decision
Pilot criterion not met: natural repeats are absent (0 retries) or deterministic/artefactual. Per the
agreed rule, **no R7 campaign is run**; creating repeats by lengthening stalls until timers expire would
manufacture the phenomenon. The paper states this as a limitation and as future work on a production-like
pipeline (per-UE log alerting with real repeat/grouping semantics).
