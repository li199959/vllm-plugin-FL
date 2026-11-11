"""GPU busy vs wall time per decode-only step from one rank's torch trace.

Step boundaries are the sampling kernel (one launch per engine step); steps
containing the sparse prefill attention kernel are skipped. If busy << wall,
decode is host-bound and faster kernels will not lower TPOT.
Usage: decode_idle.py <trace.json.gz> [...]
"""

import gzip
import statistics
import sys

import ijson

STEP = "TopPSamplingFromProb"
PREFILL = "_gqa_sparse_fwd_kernel"


def analyze(path):
    kernels = []
    with gzip.open(path, "rb") as stream:
        for ev in ijson.items(stream, "traceEvents.item", use_float=True):
            if ev.get("cat") == "kernel":
                kernels.append((ev["ts"], ev.get("dur", 0.0), ev["name"]))
    kernels.sort()
    marks = [ts for ts, _, name in kernels if STEP in name]
    walls, busys, gaps = [], [], []
    j = 0
    for a, b in zip(marks, marks[1:]):
        while j < len(kernels) and kernels[j][0] < a:
            j += 1
        k, step = j, []
        while k < len(kernels) and kernels[k][0] < b:
            step.append(kernels[k])
            k += 1
        if any(PREFILL in name for _, _, name in step):
            continue
        busy, end = 0.0, a
        largest = 0.0
        for ts, dur, _ in step:
            start = max(ts, end)
            largest = max(largest, ts - end)
            if ts + dur > start:
                busy += ts + dur - start
                end = ts + dur
        walls.append(b - a)
        busys.append(busy)
        gaps.append(largest)
    wall, busy = statistics.median(walls), statistics.median(busys)
    print(
        f"{path.split('/')[-3]}: decode steps {len(walls)}  wall {wall / 1e3:.2f} ms  "
        f"GPU busy {busy / 1e3:.2f} ms ({100 * busy / wall:.0f}%)  "
        f"largest idle gap median {statistics.median(gaps) / 1e3:.2f} ms"
    )


def collect(path):
    """Per-kernel (launches, us) per decode-only step."""
    kernels = []
    with gzip.open(path, "rb") as stream:
        for ev in ijson.items(stream, "traceEvents.item", use_float=True):
            if ev.get("cat") == "kernel":
                kernels.append((ev["ts"], ev.get("dur", 0.0), ev["name"]))
    kernels.sort()
    marks = [ts for ts, _, name in kernels if STEP in name]
    totals, steps, j = {}, 0, 0
    for a, b in zip(marks, marks[1:]):
        while j < len(kernels) and kernels[j][0] < a:
            j += 1
        k, step = j, []
        while k < len(kernels) and kernels[k][0] < b:
            step.append(kernels[k])
            k += 1
        if any(PREFILL in name for _, _, name in step):
            continue
        steps += 1
        for _, dur, name in step:
            row = totals.setdefault(name, [0, 0.0])
            row[0] += 1
            row[1] += dur
    return {n: (c / steps, t / steps) for n, (c, t) in totals.items()}


if __name__ == "__main__":
    per_kernel = None
    for p in sys.argv[1:]:
        analyze(p)
    if len(sys.argv) == 3:
        a, b = collect(sys.argv[1]), collect(sys.argv[2])
        rows = sorted(
            ((b.get(n, (0, 0))[1] - a.get(n, (0, 0))[1], n) for n in set(a) | set(b)),
            key=lambda r: -abs(r[0]),
        )
        print(
            f"{'decode-only kernel':60s} {'A n':>6s} {'A us':>8s} {'B n':>6s} {'B us':>8s} {'B-A':>7s}"
        )
        for d, n in rows[:30]:
            ca, ta = a.get(n, (0, 0.0))
            cb, tb = b.get(n, (0, 0.0))
            print(f"{n[:60]:60s} {ca:6.1f} {ta:8.1f} {cb:6.1f} {tb:8.1f} {d:+7.1f}")
