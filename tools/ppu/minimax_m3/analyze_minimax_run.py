"""Analyze completed worker traces using the installed coverage skill."""

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument(
    "--skill-dir", type=Path, default=Path("/root/.codex/skills/vllm-gpu-coverage")
)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--bench-dir", default="bench")
args = parser.parse_args()
run = args.run_dir.resolve()
scripts = args.skill_dir / "scripts"
traces = []
for p in run.glob("*.pt.trace.json.gz"):
    match = re.search(r"_rank(\d+)\.", p.name)
    if match:
        traces.append((int(match[1]), p))
audits = sorted(run.glob("worker_audit_rank*.json"))
expected = {json.loads(p.read_text())["rank"] for p in audits}
assert {r for r, _ in traces} == expected and len(traces) == len(expected) == 16


def scan(pair):
    rank, trace = pair
    out = run / f"rank{rank}.coverage_scan.json"
    if not out.exists():
        subprocess.run(
            [
                sys.executable,
                str(scripts / "scan_trace.py"),
                str(trace),
                str(out),
                "--require-graph",
            ],
            check=True,
        )
    data = json.loads(out.read_text())
    assert data["complete_json_and_gzip"] and data["graph_replay_kernel_events"] > 0
    print(
        f"rank {rank}: {len(data['kernels'])} symbols; graph events {data['graph_replay_kernel_events']}",
        flush=True,
    )
    return out


with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    scans = list(pool.map(scan, sorted(traces)))
provenance = run / "provenance.json"
if not provenance.exists():
    command = [
        sys.executable,
        str(scripts / "provenance.py"),
        "--scans",
        *map(str, scans),
    ]
    for cache in [
        run / "cache",
        Path("/root/.triton/cache"),
        Path("/root/.deep_gemm/cache"),
        Path("/tmp/torchinductor_root"),
    ]:
        if cache.is_dir():
            command += ["--cache", str(cache)]
    subprocess.run(command + ["--out", str(provenance)], check=True)
subprocess.run(
    [
        sys.executable,
        str(scripts / "coverage.py"),
        "--scans",
        *map(str, scans),
        "--audit",
        *map(str, audits),
        "--log",
        str(run / "flaggems.txt"),
        str(run / args.bench_dir / "flaggems_after_warmup.txt"),
        "--provenance",
        str(provenance),
        "--model",
        "MiniMax-M3",
        "--out",
        str(run / "gpu_coverage"),
    ],
    check=True,
)
