#!/usr/bin/env python3
"""R4 campaign driver — runs INSIDE the core container.

Modes:
  preflight            check NF processes, gNB NG Setup, one UE registration
  pilot                2 episodes per class (~4-5 min)
  campaign [N]         N episodes per class (default 30; ~4 x 30 x 32 s ≈ 65 min)
Each episode: tcpdump on lo (NGAP/SCTP + SBI HTTP/2), Prometheus scrapes every 0.5 s, NF log byte
offsets, UE launcher (nr-ue), fault injector; everything recorded in /out/episodes/ep_XXXX/meta.json.
Classes: healthy, amf_stall, ausf_stall (AUSF+UDM), reg_reject (unprovisioned IMSIs).
"""
import glob, json, os, random, signal, subprocess, sys, threading, time, urllib.request
import yaml

OUT = "/out"
LOGDIR = "/opt/open5gs/var/log/open5gs"
CFGDIR = "/opt/open5gs/etc/open5gs"
UE_BIN = "/opt/ueransim/build/nr-ue"
UE_CFG = "/opt/ueransim/config/open5gs-ue.yaml"
PROV_BASE = 999700000000001      # provisioned (see provision.js)
UNPROV_BASE = 999700000900001    # never provisioned
CLASSES = ["healthy", "amf_stall", "ausf_stall", "reg_reject"]
CLASSES_R5 = ["healthy", "amf_stall", "ausf_stall", "reg_reject", "net_delay"]
AUSF_IP = "127.0.0.11"
N_UE, UE_SECONDS, COOLDOWN, PERIOD = 20, 20.0, 5.0, 0.4
STATE = os.path.join(OUT, "campaign_state.json")


def pid_of(nf):
    try:
        return int(open(f"/tmp/{nf}.pid").read().strip())
    except Exception:
        return None


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except Exception:
        return False


def metric_endpoints():
    eps = []
    for f in sorted(glob.glob(os.path.join(CFGDIR, "*.yaml"))):
        try:
            y = yaml.safe_load(open(f)) or {}
        except Exception:
            continue
        for nf, sec in y.items():
            if nf not in ("amf", "smf", "pcf", "upf"):   # 5G SA NFs that expose Prometheus metrics
                continue
            if isinstance(sec, dict) and isinstance(sec.get("metrics"), dict):
                for s in sec["metrics"].get("server", []) or []:
                    eps.append((nf, f"http://{s['address']}:{s.get('port', 9090)}/metrics"))
    return eps


def log_offsets():
    return {os.path.basename(p): os.path.getsize(p) for p in glob.glob(os.path.join(LOGDIR, "*.log"))}


class Scraper(threading.Thread):
    def __init__(self, path, eps):
        super().__init__(daemon=True); self.path, self.eps, self.stop = path, eps, threading.Event()

    def run(self):
        with open(self.path, "w") as f:
            while not self.stop.is_set():
                t = time.time()
                for nf, url in self.eps:
                    try:
                        txt = urllib.request.urlopen(url, timeout=0.3).read().decode()
                        lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
                        f.write(json.dumps({"t": t, "nf": nf, "lines": lines}) + "\n")
                    except Exception as e:
                        f.write(json.dumps({"t": t, "nf": nf, "error": str(e)[:80]}) + "\n")
                f.flush()
                self.stop.wait(max(0.0, 0.5 - (time.time() - t)))


class Staller(threading.Thread):
    """Duty-cycled SIGSTOP/SIGCONT of NF processes: processing delay without killing the NF."""
    def __init__(self, pids, duty, period=None):
        super().__init__(daemon=True); self.pids, self.duty, self.stop = pids, duty, threading.Event()
        self.period = period or PERIOD
        self.log = []

    def run(self):
        while not self.stop.is_set():
            t0 = time.time()
            try:
                for p in self.pids:
                    os.kill(p, signal.SIGSTOP)
            except ProcessLookupError:
                self.log.append(("nf_gone", time.time())); break
            time.sleep(self.duty * self.period)
            for p in self.pids:
                os.kill(p, signal.SIGCONT)
            self.log.append((t0, time.time()))
            self.stop.wait((1 - self.duty) * self.period)
        for p in self.pids:
            try:
                os.kill(p, signal.SIGCONT)
            except Exception:
                pass


def launch_ues(ep_dir, imsi_start, n, tag):
    out = open(os.path.join(ep_dir, f"ue_{tag}.log"), "w")
    return subprocess.Popen([UE_BIN, "-c", UE_CFG, "-i", f"imsi-{imsi_start}", "-n", str(n), "-t", "100"],
                            stdout=out, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def stop_proc(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=3)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass


def netem(on, delay_ms=0):
    """Delay every packet sent to one NF address on the loopback interface.
    Tries a prio-based tree, then an htb-based tree (kernels differ in which classful qdisc is built in)."""
    subprocess.run(["tc", "qdisc", "del", "dev", "lo", "root"], stderr=subprocess.DEVNULL)
    if not on:
        return True
    match = ["match", "ip", "dst", f"{AUSF_IP}/32"]
    trees = [
        [["tc", "qdisc", "add", "dev", "lo", "root", "handle", "1:", "prio"],
         ["tc", "qdisc", "add", "dev", "lo", "parent", "1:3", "handle", "30:", "netem", "delay", f"{delay_ms}ms"],
         ["tc", "filter", "add", "dev", "lo", "protocol", "ip", "parent", "1:0", "prio", "3", "u32", *match, "flowid", "1:3"]],
        [["tc", "qdisc", "add", "dev", "lo", "root", "handle", "1:", "htb", "default", "10"],
         ["tc", "class", "add", "dev", "lo", "parent", "1:", "classid", "1:10", "htb", "rate", "10gbit"],
         ["tc", "class", "add", "dev", "lo", "parent", "1:", "classid", "1:30", "htb", "rate", "10gbit"],
         ["tc", "qdisc", "add", "dev", "lo", "parent", "1:30", "handle", "30:", "netem", "delay", f"{delay_ms}ms"],
         ["tc", "filter", "add", "dev", "lo", "protocol", "ip", "parent", "1:", "prio", "1", "u32", *match, "flowid", "1:30"]],
    ]
    for cmds in trees:
        if all(subprocess.run(c, stderr=subprocess.DEVNULL).returncode == 0 for c in cmds):
            return True
        subprocess.run(["tc", "qdisc", "del", "dev", "lo", "root"], stderr=subprocess.DEVNULL)
    return False


def run_episode(idx, label, intensity, imsi_start, eps, period=None, block=None, sched_pos=None):
    ep_dir = os.path.join(OUT, "episodes", f"ep_{idx:04d}")
    os.makedirs(ep_dir, exist_ok=True)
    meta = {"episode": idx, "label": label, "intensity": intensity, "imsi_start": imsi_start, "n_ue": N_UE,
            "period_s": period or PERIOD, "ue_seconds": UE_SECONDS, "block": block, "sched_pos": sched_pos}
    cap = subprocess.Popen(["tcpdump", "-i", "lo", "-U", "-w", os.path.join(ep_dir, "capture.pcap"),
                            "sctp or tcp port 7777"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sc = Scraper(os.path.join(ep_dir, "metrics.jsonl"), eps); sc.start()
    time.sleep(1.0)
    meta["log_offsets_start"] = log_offsets(); meta["t_start"] = time.time()
    staller = None
    if label in ("amf_stall", "ausf_stall"):
        nfs = ["amf"] if label == "amf_stall" else ["ausf", "udm"]
        pids = [pid_of(nf) for nf in nfs]
        staller = Staller([p for p in pids if p], intensity, period); staller.start()
        meta["stalled_nfs"] = nfs
    if label == "net_delay":
        meta["netem_ok"] = netem(True, int(intensity))
        meta["netem_target"] = AUSF_IP
        meta["tc_show"] = subprocess.run(["tc", "qdisc", "show", "dev", "lo"], capture_output=True, text=True).stdout
    procs = []
    if label == "reg_reject":
        k = max(1, round(intensity * N_UE))
        procs.append(launch_ues(ep_dir, imsi_start, N_UE - k, "prov"))
        procs.append(launch_ues(ep_dir, UNPROV_BASE + idx * N_UE, k, "unprov"))
        meta["n_unprovisioned"] = k
    else:
        procs.append(launch_ues(ep_dir, imsi_start, N_UE, "prov"))
    time.sleep(UE_SECONDS)
    for p in procs:
        stop_proc(p)
    if staller:
        staller.stop.set(); staller.join(timeout=2); meta["stall_intervals"] = staller.log
    if label == "net_delay":
        netem(False)
    time.sleep(2.0)
    meta["t_end"] = time.time(); meta["log_offsets_end"] = log_offsets()
    sc.stop.set(); sc.join(timeout=2)
    cap.send_signal(signal.SIGINT)
    try:
        cap.wait(timeout=5)
    except Exception:
        cap.kill()
    meta["nf_alive_after"] = {nf: bool(pid_of(nf) and alive(pid_of(nf))) for nf in
                              ["nrf", "scp", "amf", "smf", "upf", "ausf", "udm", "pcf", "nssf", "bsf", "udr", "gnb"]}
    json.dump(meta, open(os.path.join(ep_dir, "meta.json"), "w"), indent=1)
    return meta


def preflight():
    ok = True
    for nf in ["nrf", "scp", "amf", "smf", "upf", "ausf", "udm", "pcf", "nssf", "bsf", "udr", "gnb"]:
        p = pid_of(nf); a = bool(p and alive(p)); ok &= a
        print(f"{nf:5s} {'running' if a else 'NOT RUNNING'}")
    g = open("/out/logs/gnb.stdout").read() if os.path.exists("/out/logs/gnb.stdout") else ""
    ng = "NG Setup procedure is successful" in g
    print("gNB NG Setup:", "OK" if ng else "NOT OK (check SCTP support in the Docker VM)"); ok &= ng
    print("metrics endpoints:", metric_endpoints())
    os.makedirs("/out/preflight", exist_ok=True)
    p = launch_ues("/out/preflight", PROV_BASE, 1, "single"); time.sleep(8); stop_proc(p)
    txt = open("/out/preflight/ue_single.log").read()
    reg = "Registration is successful" in txt or "registration is successful" in txt.lower()
    print("UE registration:", "OK" if reg else "NOT OK (see out/preflight/ue_single.log; subscribers provisioned?)")
    ok &= reg
    print("PREFLIGHT", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def core_ready():
    g = open("/out/logs/gnb.stdout").read() if os.path.exists("/out/logs/gnb.stdout") else ""
    nfs_ok = all(pid_of(nf) and alive(pid_of(nf)) for nf in ["nrf", "scp", "amf", "smf", "ausf", "udm", "udr", "gnb"])
    return nfs_ok and "NG Setup procedure is successful" in g


def campaign(n_per_class, seed=20261023, state_path=None):
    state_path = state_path or STATE
    if not core_ready() and os.environ.get("PAEF_FORCE") != "1":
        print("ABORT: core or gNB not ready. Run ./run.sh preflight and fix it first.")
        sys.exit(2)
    rng = random.Random(seed)
    sched = [c for c in CLASSES for _ in range(n_per_class)]
    rng.shuffle(sched)
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"done": 0, "next_idx": 0}
    # episode indices continue across pilot and campaign so directories never collide
    if state["next_idx"] == 0:
        existing = glob.glob(os.path.join(OUT, "episodes", "ep_*"))
        state["next_idx"] = len(existing)
    eps = metric_endpoints()
    t0 = time.time()
    for i, label in enumerate(sched):
        if i < state["done"]:
            continue  # resumable
        intensity = 0.0 if label == "healthy" else (rng.uniform(0.05, 0.40) if "stall" in label else rng.uniform(0.05, 0.5))
        idx = state["next_idx"]
        if not core_ready() and os.environ.get("PAEF_FORCE") != "1":
            print(f"ABORT before episode {idx}: an NF or the gNB stopped. Episodes so far are kept; "
                  "restart with ./run.sh down && ./run.sh up, then rerun the same command (it resumes).")
            sys.exit(3)
        m = run_episode(idx, label, round(intensity, 3), PROV_BASE + 1000 + idx * N_UE, eps)
        state["done"] = i + 1; state["next_idx"] = idx + 1
        json.dump(state, open(state_path, "w"))
        el = time.time() - t0
        print(f"[{i + 1}/{len(sched)}] ep_{idx:04d} {label:10s} intensity={m['intensity']:.2f} "
              f"elapsed={el / 60:.1f} min", flush=True)
        time.sleep(COOLDOWN)
    print("CAMPAIGN DONE")


def r5_block(block, n_per_class):
    """One independent R5 block (the core was restarted and re-provisioned just before)."""
    state_path = os.path.join(OUT, f"r5_block{block}_state.json")
    if not core_ready() and os.environ.get("PAEF_FORCE") != "1":
        print("ABORT: core or gNB not ready."); sys.exit(2)
    net_ok = netem(True, 10)
    netem(False)
    classes = CLASSES_R5 if net_ok else [c for c in CLASSES_R5 if c != "net_delay"]
    if not net_ok:
        print("WARNING: tc netem unavailable in this Docker VM -> net_delay dropped for this block (pre-registered fallback).")
    rng = random.Random(20261025 + 1000 * block)
    sched = [c for c in classes for _ in range(n_per_class)]
    rng.shuffle(sched)

    def draw(label):
        if label == "healthy":
            return 0.0
        if label in ("amf_stall", "ausf_stall"):
            return rng.uniform(0.30, 0.70)
        if label == "reg_reject":
            return rng.uniform(0.05, 0.50)
        return rng.uniform(30, 200)  # net_delay: one-way delay in ms to the AUSF address
    plan = [(label, draw(label)) for label in sched]   # drawn up front: resuming does not change the plan
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"done": 0}
    state["net_delay_available"] = net_ok
    eps = metric_endpoints(); t0 = time.time()
    for i, (label, inten) in enumerate(plan):
        if i < state["done"]:
            continue
        if not core_ready() and os.environ.get("PAEF_FORCE") != "1":
            print(f"ABORT in block {block}: an NF or the gNB stopped. Rerun the same command to resume."); sys.exit(3)
        idx = len(glob.glob(os.path.join(OUT, "episodes", "ep_*")))
        m = run_episode(idx, label, round(inten, 3), PROV_BASE + 200 + idx * N_UE, eps, period=1.0, block=block, sched_pos=i)
        state["done"] = i + 1
        json.dump(state, open(state_path, "w"))
        print(f"[block {block}: {i + 1}/{len(sched)}] ep_{idx:04d} {label:10s} intensity={m['intensity']:.2f} "
              f"elapsed={(time.time() - t0) / 60:.1f} min", flush=True)
        time.sleep(COOLDOWN)
    state["complete"] = True
    json.dump(state, open(state_path, "w"))
    print(f"BLOCK {block} DONE")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    if mode == "preflight":
        sys.exit(preflight())
    elif mode == "pilot":
        campaign(2, seed=7, state_path=os.path.join(OUT, "pilot_state.json"))
    elif mode == "r5block":
        r5_block(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    elif mode == "campaign":
        campaign(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
