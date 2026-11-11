"""Read-only progress summary for a captured MiniMax run."""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
run = args.run_dir
for name in (
    "server.log",
    "bench/warmup1.log",
    "bench/warmup2.log",
    "bench/profile.log",
):
    p = run / name
    if p.exists():
        lines = p.read_text(errors="replace").splitlines()
        print(name, "bytes=", p.stat().st_size)
        print("\n".join(lines[-5:]))
for p in sorted((run / "bench").glob("*.json")):
    if "command" not in p.name:
        data = json.loads(p.read_text())
        print(
            p.name,
            {
                k: data[k]
                for k in (
                    "completed",
                    "failed",
                    "duration",
                    "total_input_tokens",
                    "total_output_tokens",
                    "output_throughput",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                )
                if k in data
            },
        )
print("traces", len(list(run.glob("*.pt.trace.json.gz"))))
try:
    with urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=3) as r:
        print("health", r.status)
except Exception as exc:
    print("health", type(exc).__name__)
processes = subprocess.check_output(
    ["ps", "-eo", "pid,ppid,etime,%cpu,stat,cmd"], text=True
)
lines = processes.splitlines()
compilers = [line for line in lines if "ppu-llc" in line or "ptxas" in line]
print("compiler processes", len(compilers))
for line in compilers[:2]:
    print(line[:250])
