"""Summarize observed linear shapes and GPU launch geometry from one trace."""

import argparse
import collections
import gzip
import json
from pathlib import Path

import ijson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    scan = json.loads((args.run / "rank0.coverage_scan.json").read_text())
    shapes = collections.Counter()
    launches = {}
    with gzip.open(scan["trace"], "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            name, cat, meta = event.get("name"), event.get("cat"), event.get("args", {})
            if cat == "cpu_op" and name in ("aten::linear", "aten::mm", "aten::addmm"):
                shapes[
                    (
                        name,
                        json.dumps(meta.get("Input Dims")),
                        json.dumps(meta.get("Input type")),
                    )
                ] += 1
            if cat == "kernel" and name in ("linear_kernel", "fused_moe_kernel"):
                key = (
                    name,
                    str(meta.get("grid")),
                    str(meta.get("block")),
                    meta.get("registers per thread"),
                    meta.get("shared memory"),
                )
                row = launches.setdefault(key, {"count": 0, "duration_us": 0})
                row["count"] += 1
                row["duration_us"] += event.get("dur", 0)
    result = {
        "shapes": [
            {
                "op": key[0],
                "dims": json.loads(key[1]),
                "types": json.loads(key[2]),
                "count": value,
            }
            for key, value in shapes.most_common()
        ],
        "launches": [
            {"signature": key, **value}
            for key, value in sorted(
                launches.items(), key=lambda kv: -kv[1]["duration_us"]
            )
        ],
    }
    output = args.run / "shape_performance.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
