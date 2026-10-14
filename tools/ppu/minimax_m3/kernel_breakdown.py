"""Per-decode-step GPU kernel time from one rank's torch trace, and a diff.

Usage: kernel_breakdown.py <trace.json.gz> [<other trace.json.gz>]
Steps are counted from the sparse-attention decode kernel (one per layer).
"""
import collections
import gzip
import json
import sys

import ijson

LAYERS = 60
STEP_KERNEL = "_gqa_sparse_decode_kernel"


def load(path):
    totals = collections.defaultdict(lambda: [0, 0.0])
    with gzip.open(path, "rb") as stream:
        for ev in ijson.items(stream, "traceEvents.item", use_float=True):
            if ev.get("cat") == "kernel":
                row = totals[ev["name"]]
                row[0] += 1
                row[1] += ev.get("dur", 0.0)
    steps = sum(c for n, (c, _) in totals.items() if STEP_KERNEL in n) / LAYERS
    return {n: (c / steps, t / steps) for n, (c, t) in totals.items()}, steps


def main():
    a, sa = load(sys.argv[1])
    print(f"steps: {sa:.0f}  total {sum(t for _, t in a.values()):.0f} us/step  "
          f"{sum(c for c, _ in a.values()):.0f} launches/step")
    if len(sys.argv) < 3:
        for n, (c, t) in sorted(a.items(), key=lambda kv: -kv[1][1])[:30]:
            print(f"{n[:70]:70s} {c:7.1f}/step {t:9.1f} us/step")
        return
    b, sb = load(sys.argv[2])
    print(f"steps: {sb:.0f}  total {sum(t for _, t in b.values()):.0f} us/step  "
          f"{sum(c for c, _ in b.values()):.0f} launches/step  (second trace)")
    rows = []
    for n in set(a) | set(b):
        ca, ta = a.get(n, (0, 0.0))
        cb, tb = b.get(n, (0, 0.0))
        rows.append((tb - ta, n, ca, ta, cb, tb))
    print(f"{'kernel':62s} {'A n':>6s} {'A us':>8s} {'B n':>6s} {'B us':>8s} {'B-A':>8s}")
    for d, n, ca, ta, cb, tb in sorted(rows, key=lambda r: -abs(r[0]))[:35]:
        print(f"{n[:62]:62s} {ca:6.1f} {ta:8.1f} {cb:6.1f} {tb:8.1f} {d:+8.1f}")


if __name__ == "__main__":
    main()
