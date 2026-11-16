#!/usr/bin/env python3
"""R11 driver (runs INSIDE the core container). Reuses run_episode() from campaign.py unchanged; the R10 files
are not modified.

Modes:
  loadpilot              technical pilot of the load manipulation (NOT used for any hypothesis test):
                         sub-runs alternating load levels 0 / half / full, each 4 healthy + 1 amf_stall + 1 ausf_stall
  r11block <b> <n> <W>   one confirmatory block: n episodes per class per condition (control: 0 workers,
                         loaded: W workers). The condition is assigned per episode by a balanced random schedule
                         fixed in advance from the block seed, independent of the class.
Each episode's meta.json gets: load_workers, load_cond, and a manipulation check read from /proc of the
Docker VM (loadavg at start/end, CPU busy fraction during the episode).
IMSIs use the block-local index (the subscriber database is wiped and re-provisioned for every block), so
no block can run past the provisioned range."""
import json, os, random, sys, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from campaign import (OUT, PROV_BASE, N_UE, COOLDOWN, CLASSES_R5, run_episode, core_ready,  # noqa: E402
                      metric_endpoints, netem)

CTL = os.path.join(OUT, "load_ctl")
WARMUP = 4.0


def cpu_times():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]          # total, idle + iowait


def set_load(w):
    open(CTL, "w").write(str(int(w)))


def draw(label, rng):
    if label == "healthy":
        return 0.0
    if label in ("amf_stall", "ausf_stall"):
        return rng.uniform(0.30, 0.70)
    if label == "reg_reject":
        return rng.uniform(0.05, 0.50)
    return rng.uniform(30, 200)


def one(i_local, label, inten, workers, cond, block, pos, eps):
    set_load(workers); time.sleep(WARMUP if workers else 1.0)
    la0 = open("/proc/loadavg").read().split()[:3]; t0, i0 = cpu_times()
    idx = len(glob.glob(os.path.join(OUT, "episodes", "ep_*")))
    m = run_episode(idx, label, round(inten, 3), PROV_BASE + 200 + i_local * N_UE, eps, period=1.0,
                    block=block, sched_pos=pos)
    t1, i1 = cpu_times(); la1 = open("/proc/loadavg").read().split()[:3]
    set_load(0)
    p = os.path.join(OUT, "episodes", f"ep_{idx:04d}", "meta.json")
    m.update(load_workers=workers, load_cond=cond, loadavg_start=la0, loadavg_end=la1,
             vm_cpu_busy=round(1 - (i1 - i0) / max(1, t1 - t0), 3), vm_ncpu=os.cpu_count())
    json.dump(m, open(p, "w"), indent=1)
    return idx, m


def loadpilot():
    ncpu = os.cpu_count(); levels = [0, max(1, ncpu // 2), ncpu]
    order = [0, 1, 2, 2, 1, 0]                # alternating, balanced over time
    rng = random.Random(20261111); eps = metric_endpoints(); k = 0
    st = os.path.join(OUT, "r11_pilot_state.json")
    done = json.load(open(st))["done"] if os.path.exists(st) else 0
    plan = []
    for s, lv in enumerate(order):
        labels = ["healthy"] * 4 + ["amf_stall", "ausf_stall"]; rng.shuffle(labels)
        plan += [(s, lv, lab, draw(lab, rng)) for lab in labels]
    for j, (s, lv, lab, inten) in enumerate(plan):
        if j < done:
            continue
        if not core_ready():
            print("ABORT: core not ready (NF or gNB stopped) — this is itself a pilot finding; rerun to resume."); sys.exit(3)
        idx, m = one(j, lab, inten, levels[lv], f"pilot_L{lv}", 900 + s, j, eps)
        json.dump({"done": j + 1}, open(st, "w"))
        print(f"[pilot {j + 1}/{len(plan)}] ep_{idx:04d} L{lv} workers={levels[lv]} {lab:10s} "
              f"vm_cpu_busy={m['vm_cpu_busy']:.2f} alive={all(m['nf_alive_after'].values())}", flush=True)
        time.sleep(COOLDOWN)
    set_load(0); print("LOADPILOT DONE")


def r11block(block, n, W):
    if not core_ready():
        print("ABORT: core or gNB not ready."); sys.exit(2)
    if not netem(True, 10):
        print("ABORT: tc netem unavailable; R11 requires net_delay."); sys.exit(2)
    netem(False)
    rng = random.Random(20261201 + 1000 * block)
    jobs = [(lab, cond) for lab in CLASSES_R5 for cond in ("control", "loaded") for _ in range(n)]
    rng.shuffle(jobs)
    plan = [(lab, cond, draw(lab, rng)) for lab, cond in jobs]
    st = os.path.join(OUT, f"r11_block{block}_state.json")
    state = json.load(open(st)) if os.path.exists(st) else {"done": 0}
    eps = metric_endpoints(); t0 = time.time()
    for i, (lab, cond, inten) in enumerate(plan):
        if i < state["done"]:
            continue
        if not core_ready():
            print(f"ABORT in block {block}: an NF or the gNB stopped. Rerun the same command to resume."); sys.exit(3)
        idx, m = one(i, lab, inten, W if cond == "loaded" else 0, cond, block, i, eps)
        state["done"] = i + 1; json.dump(state, open(st, "w"))
        print(f"[R11 block {block}: {i + 1}/{len(plan)}] ep_{idx:04d} {cond:7s} {lab:10s} "
              f"cpu={m['vm_cpu_busy']:.2f} elapsed={(time.time() - t0) / 60:.1f} min", flush=True)
        time.sleep(COOLDOWN)
    state["complete"] = True; json.dump(state, open(st, "w")); set_load(0)
    print(f"R11 BLOCK {block} DONE")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "loadpilot":
        loadpilot()
    elif mode == "r11block":
        r11block(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        print(__doc__)
