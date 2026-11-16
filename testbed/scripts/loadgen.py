#!/usr/bin/env python3
"""R11 background-load sidecar ("noisy neighbour"). Runs in its own container on the same Docker VM as the core.
Reads the number of busy workers from /out/load_ctl (written by campaign_r11.py before each episode) and keeps
exactly that many workers busy. Each busy worker spins on the CPU and rewrites a 64 MB buffer, so it competes
for CPU time and memory bandwidth. The load is set per episode by a schedule fixed in advance; it never
depends on the episode's class."""
import multiprocessing as mp, os, time

CTL = "/out/load_ctl"
MAXW = int(os.environ.get("LOAD_MAXW", "16"))


def worker(flag):
    buf = bytearray(64 * 1024 * 1024)
    i = 0
    while True:
        if flag.value:
            for j in range(0, len(buf), 4096):      # memory traffic
                buf[j] = (buf[j] + 1) & 0xFF
            x = 0
            for k in range(200000):                 # CPU spin
                x += k * k
            i += 1
        else:
            time.sleep(0.05)


def main():
    flags = [mp.Value("b", 0) for _ in range(MAXW)]
    for f in flags:
        mp.Process(target=worker, args=(f,), daemon=True).start()
    last = None
    while True:
        try:
            n = max(0, min(MAXW, int(open(CTL).read().strip() or 0)))
        except Exception:
            n = 0
        if n != last:
            for k, f in enumerate(flags):
                f.value = 1 if k < n else 0
            print(f"{time.time():.1f} busy_workers={n}", flush=True); last = n
        time.sleep(0.2)


if __name__ == "__main__":
    main()
