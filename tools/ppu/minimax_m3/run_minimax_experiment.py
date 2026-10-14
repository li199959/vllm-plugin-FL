"""Own one M3 comparison: startup, identical benchmark, profiler drain, stop."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="/mnt/cpfs/xiangbin/MiniMax-M3")
    parser.add_argument("--moe", default="hybrid")
    parser.add_argument("--dense", default="hybrid")
    parser.add_argument("--norm", default="flaggems")
    parser.add_argument("--fused", default="triton")
    parser.add_argument("--index", default="native")  # Gems index costs ~1 GiB KV, see README
    parser.add_argument("--stable-rounds", type=int, default=0,
                        help="Additional identical rounds after profiler drain, without profiling")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    run = args.run_dir.resolve()
    command = [sys.executable, str(root / "tools/ppu/minimax_m3/serve_minimax_m3_gems.py"),
               "--model", args.model, "--run-dir", str(run), "--profile", "--log-file"]
    for name in ("moe", "dense", "norm", "fused", "index"):
        command += ["--" + name, getattr(args, name)]
    server = subprocess.Popen(command, cwd=root)
    try:
        start = time.monotonic()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited: {server.returncode}; inspect {run / 'server.log'}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() - start > 3600:
                raise TimeoutError("Server startup exceeded one hour")
            print(f"Waiting for startup: {time.monotonic() - start:.0f}s", flush=True)
            time.sleep(15)
        subprocess.run([
            sys.executable, "/root/.codex/skills/vllm-gpu-coverage/scripts/run_benchmark.py",
            "--command-json", str(run / "bench.json"), "--out", str(run / "bench"),
            "--base-url", "http://127.0.0.1:8100", "--flaggems-log", str(run / "flaggems.txt"),
            "--expected-prompts", "128", "--expected-input-tokens", "524288",
            "--expected-output-tokens", "131072",
        ], check=True)
        start = time.monotonic()
        while '"POST /stop_profile HTTP/1.1" 200' not in (run / "server.log").read_text(errors="replace"):
            if time.monotonic() - start > 900:
                raise TimeoutError("Profiler did not confirm drain")
            print("Waiting for profiler drain", flush=True)
            time.sleep(15)
        summary = json.loads((run / "bench/profile.json").read_text())
        print(json.dumps({k: summary[k] for k in ("completed", "failed", "output_throughput", "mean_ttft_ms", "mean_tpot_ms")}), flush=True)
        benchmark = json.loads((run / "bench.json").read_text())
        for index in range(1, args.stable_rounds + 1):
            name = f"stable{index}"
            command = benchmark + ["--save-result", "--save-detailed", "--result-dir",
                                   str(run / "bench"), "--result-filename", name + ".json"]
            (run / (name + ".argv.json")).write_text(json.dumps(command, indent=2) + "\n")
            print(name + " starting without profiler", flush=True)
            with (run / "bench" / (name + ".log")).open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            data = json.loads((run / "bench" / (name + ".json")).read_text())
            assert data["completed"] == 128 and data["failed"] == 0
            assert data["total_input_tokens"] == 524288 and data["total_output_tokens"] == 131072
            print(json.dumps({"round": name, **{k: data[k] for k in
                  ("output_throughput", "mean_ttft_ms", "mean_tpot_ms")}}), flush=True)
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            for _ in range(60):
                if server.poll() is not None:
                    break
                time.sleep(1)
            if server.poll() is None:
                raise RuntimeError(f"Owned server {server.pid} did not stop after SIGINT")


if __name__ == "__main__":
    main()
