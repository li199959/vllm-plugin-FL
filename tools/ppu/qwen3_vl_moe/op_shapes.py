#!/usr/bin/env python3
"""Count CPU-op calls in a torch profiler trace by input dims/strides/dtypes.

Used to find which call site feeds a Gems kernel with ever-changing strides
(pointwise_dynamic bakes strides in as constexpr, so each one recompiles).
Needs a trace captured with record_shapes=True.

  python3 op_shapes.py TRACE.pt.trace.json.gz --op aten::copy_ --top 20
"""

import argparse
import collections
import gzip

import ijson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--op", default="aten::copy_")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    opener = gzip.open if a.trace.endswith(".gz") else open
    counts = collections.Counter()
    with opener(a.trace, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("name") != a.op or ev.get("cat") != "cpu_op":
                continue
            args = ev.get("args", {})
            key = (
                str(args.get("Input type")),
                str(args.get("Input Dims")),
                str(args.get("Input Strides")),
            )
            counts[key] += 1
    print(f"{a.op}: {sum(counts.values())} calls, {len(counts)} distinct signatures")
    # Group by dtype+strides with sizes dropped, to show stride churn per site.
    for (types, dims, strides), n in counts.most_common(a.top):
        print(f"{n:7d}  types={types} dims={dims} strides={strides}")


if __name__ == "__main__":
    main()
