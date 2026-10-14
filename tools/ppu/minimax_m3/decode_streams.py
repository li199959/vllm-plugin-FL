"""Per-stream kernel time in decode-only steps (which work overlaps MoE?).

Usage: decode_streams.py <trace.json.gz>
"""
import collections
import gzip
import sys

import ijson

STEP = "TopPSamplingFromProb"
PREFILL = "_gqa_sparse_fwd_kernel"


def main(path):
    kernels = []
    with gzip.open(path, "rb") as stream:
        for ev in ijson.items(stream, "traceEvents.item", use_float=True):
            if ev.get("cat") == "kernel":
                kernels.append((ev["ts"], ev.get("dur", 0.0), ev["name"],
                                ev.get("args", {}).get("stream")))
    kernels.sort()
    marks = [ts for ts, _, name, _ in kernels if STEP in name]
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    steps, j = 0, 0
    for a, b in zip(marks, marks[1:]):
        while j < len(kernels) and kernels[j][0] < a:
            j += 1
        k, step = j, []
        while k < len(kernels) and kernels[k][0] < b:
            step.append(kernels[k])
            k += 1
        if any(PREFILL in name for _, _, name, _ in step):
            continue
        steps += 1
        for _, dur, name, sid in step:
            per[sid][name] += dur
    print(path.split("/")[-3], "decode steps", steps)
    for sid, names in sorted(per.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(names.values()) / steps
        top = sorted(names.items(), key=lambda kv: -kv[1])[:int(__import__("os").environ.get("TOPN", "6"))]
        print(f"  stream {sid}: {total / 1e3:.2f} ms/step")
        for name, t in top:
            print(f"      {t / steps:8.1f} us  {name[:70]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
