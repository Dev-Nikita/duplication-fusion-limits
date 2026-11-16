# R4 testbed — how to run it on the MacBook (Apple silicon, Docker Desktop)

Everything runs **on the Mac**, inside Docker Desktop. Commands are typed in the macOS **Terminal**
(not in a remote VM). The project folder is:
`~/Science Article 2025-/IF-Telecom-evidence-fusion/testbed`

## 0. One-time settings (2 minutes)
* Docker Desktop → Settings → Resources: **CPUs 4**, **Memory 6–8 GB**, Disk ≥ 20 GB free.
* Docker Desktop → Settings → Resources → Advanced: turn **Resource Saver** off for the campaign.
* Mac on the **power adapter**.

## 1. Build the image (once; ~20–30 min on an M1, then cached)
```bash
cd ~/"Science Article 2025-/IF-Telecom-evidence-fusion/testbed"
./run.sh build
```
Compiles Open5GS v2.7.6 and UERANSIM v3.2.7 from source for arm64 (verified build: Open5GS compile ≈ 5 min on 2 cloud cores; expect 15–25 min in total on the M1 including package download).

## 2. Start, provision, check (~3 min)
```bash
./run.sh up          # MongoDB + 5G core + gNB
./run.sh provision   # 4,000 subscribers  -> "provisioned 4000 subscribers"
./run.sh preflight   # must end with: PREFLIGHT PASSED
```
If preflight says `gNB NG Setup: NOT OK`, the Docker VM lacks SCTP — send me `out/logs/gnb.stdout`.

## 3. Pilot (~5 min) — only to finalise the parsers; excluded from the analysis
```bash
caffeinate -is ./run.sh pilot
./run.sh extract
```
Then tell me "pilot done": I read `out/episodes/` (8 pilot episodes) and freeze the parsers.

## 4. Campaign (~65–70 min for 120 episodes)
```bash
caffeinate -is ./run.sh campaign 30
./run.sh extract     # ~5–10 min
```
Progress: `./run.sh status` in a second Terminal tab, or watch the lines `[k/120] ep_XXXX ...`.

## Sleep, lid, interruptions
* `caffeinate -is` keeps the Mac awake while the command runs; the **screen may turn off** — that is fine.
* **Do not close the lid** (unless an external display, keyboard and power are connected): closing the lid
  sleeps the Mac and pauses Docker; timing-sensitive episodes would then be invalid.
* The campaign is **resumable**: if it stops (sleep, Docker restart, Ctrl-C), run the same command again;
  finished episodes are kept (`out/campaign_state.json`). The episode that was interrupted is re-run with a
  new index; tell me if that happened so it is excluded.
* Do not run heavy work on the Mac during the campaign (builds, video calls): it changes timing.

## 5. After the campaign
```bash
./run.sh down        # stops the containers (data stay in out/)
```
Then tell me "campaign done". I read `out/` (≈ 0.5–1.5 GB: pcaps, logs, metrics) and run the analysis.

## Where things are
| Path | Content |
|---|---|
| `out/logs/` | NF stdout, gNB log, core start time |
| `out/episodes/ep_XXXX/` | `capture.pcap`, `signalling.csv`, `metrics.jsonl`, `logs/*.log`, `ue_*.log`, `meta.json` |
| `out/campaign.log` | progress log |
| `docs/r4-protocol.md` | frozen analysis protocol |

## Reset completely
```bash
./run.sh reset       # removes containers and out/ — only if you want to start over
```

## R5 — three independent blocks (~2.5–3 h, no rebuild needed)
```bash
cd ~/"Science Article 2025-/IF-Telecom-evidence-fusion/testbed"
caffeinate -is ./run.sh r5all 20
```
Each block: restart containers → provision 12 000 subscribers → preflight → 80–100 episodes (20 per class).
Resumable: run the same command again; finished blocks are skipped (`out/r5_block{1,2,3}_state.json`).
Progress: `./run.sh status` in a second tab. At the end the command extracts and stops the containers.
If it prints `WARNING: tc netem unavailable`, it continues without net_delay (pre-registered fallback).
