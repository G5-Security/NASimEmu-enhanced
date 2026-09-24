"""Live progress of the follow-up evaluation jobs. Run:  watch -n 15 python3 progress.py"""
import glob
import os
import re
import subprocess
import time

jobs = []
for jf in sorted(glob.glob("jobs_*.txt")):
    for line in open(jf):
        m = re.search(r"--n_episodes (\d+).*?--out_csv (\S+)", line)
        if m:
            jobs.append((os.path.basename(m.group(2))[:-4], int(m.group(1))))


def done_rows(name):
    p = name + ".csv"
    return max(0, sum(1 for _ in open(p)) - 1) if os.path.exists(p) else 0


groups = {}
for name, n in jobs:
    groups.setdefault(name.split("_")[0], []).append((name, n, done_rows(name)))

total = sum(n for _, n in jobs)
done = sum(min(d, n) for g in groups.values() for _, n, d in g)
# rate = recent progress: compare with a sample from up to 10 minutes ago (kept in .progress_state);
# on the first run, fall back to the time since the job lists were written
now = time.time()
state_file = ".progress_state"
samples = []
if os.path.exists(state_file):
    samples = [tuple(map(float, l.split())) for l in open(state_file) if l.strip()]
samples = [(t, d) for t, d in samples if now - t < 600]
with open(state_file, "a") as f:
    f.write(f"{now} {done}\n")
if samples and now - samples[0][0] > 20:
    rate = (done - samples[0][1]) / (now - samples[0][0])
else:
    rate = done / max(1.0, now - min(os.path.getmtime(f) for f in glob.glob("jobs_*.txt")))
eta_min = (total - done) / rate / 60 if rate > 0 else float("nan")

print(time.strftime("%H:%M:%S"), f"| episodes {done}/{total} ({100 * done / total:.0f}%) | {rate:.2f} episodes/s | ETA ~{eta_min:.0f} min")
print("running processes:", subprocess.run("pgrep -fc 'experiments/eval_harness.py'", shell=True, capture_output=True, text=True).stdout.strip(),
      "| load:", open("/proc/loadavg").read().split()[0])
print()
labels = {"A": "A goal masks/traces", "B": "B IDS detections", "D": "D parameter sweep", "E": "E scan noise observed",
          "F": "F generated networks", "G": "G IDS delay", "H2": "H2 best checkpoint"}
for key, g in groups.items():
    n_done = sum(1 for _, n, d in g if d >= n)
    eps = sum(min(d, n) for _, n, d in g)
    tot = sum(n for _, n, _ in g)
    print(f"{labels.get(key, key):24s} {n_done:2d}/{len(g):2d} jobs done | {eps:5d}/{tot:5d} episodes ({100 * eps / tot:3.0f}%)")
    for name, n, d in g:
        if 0 < d < n:
            print(f"    running  {name:34s} {d:4d}/{n}")
