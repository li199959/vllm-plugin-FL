"""Measure identical M3 rounds without profiler using existing compile caches."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def healthy():
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8100/health", timeout=3
        ) as response:
            return response.status == 200
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "hybrid"), required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    plugin = (
        Path("/mnt/cpfs/xiangbin/plugin_env_0906/vllm-plugin-FL")
        if args.variant == "baseline"
        else root
    )
    reference, run = args.reference_run.resolve(), args.run_dir.resolve()
    if healthy():
        raise RuntimeError("Port 8100 already has a healthy service")
    run.mkdir(parents=True, exist_ok=False)
    (run / "bench").mkdir()
    command = json.loads((reference / "serve.argv.json").read_text())
    for option in ("--worker-extension-cls", "--profiler-config"):
        if option in command:
            index = command.index(option)
            del command[index : index + 2]
    bench_path = reference / (
        "bench.valid.json" if args.variant == "baseline" else "bench.json"
    )
    benchmark = json.loads(bench_path.read_text())
    selected = json.loads((reference / "serve.env.json").read_text())
    selected["PYTHONPATH"] = str(plugin)
    selected["FLAGGEMS_ENABLE_OPLIST_PATH"] = str(run / "flaggems.txt")
    env = os.environ.copy()
    env.update(selected)
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((plugin / "vllm_fl").rglob("*.py"))
    }
    for name, data in [
        ("serve.argv.json", command),
        ("serve.env.json", selected),
        ("bench.json", benchmark),
        ("source_hashes.json", sources),
    ]:
        (run / name).write_text(json.dumps(data, indent=2) + "\n")
    results = {}
    with (run / "server.log").open("w") as log:
        server = subprocess.Popen(
            command, cwd=plugin, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    (run / "server.pid").write_text(str(server.pid) + "\n")
    try:
        start = time.monotonic()
        while not healthy():
            if server.poll() is not None:
                raise RuntimeError(f"Server exited: {server.returncode}")
            if time.monotonic() - start > 3600:
                raise TimeoutError("Startup exceeded one hour")
            print(f"Waiting for startup: {time.monotonic() - start:.0f}s", flush=True)
            time.sleep(15)
        for name in ("warmup1", "warmup2", "stable1", "stable2"):
            if server.poll() is not None:
                raise RuntimeError("Server exited before " + name)
            argv = benchmark + [
                "--save-result",
                "--save-detailed",
                "--result-dir",
                str(run / "bench"),
                "--result-filename",
                name + ".json",
            ]
            (run / (name + ".argv.json")).write_text(json.dumps(argv, indent=2) + "\n")
            print(name + " starting without profiler", flush=True)
            with (run / "bench" / (name + ".log")).open("w") as log:
                subprocess.run(
                    argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            data = json.loads((run / "bench" / (name + ".json")).read_text())
            assert data["completed"] == 128 and data["failed"] == 0
            assert (
                data["total_input_tokens"] == 524288
                and data["total_output_tokens"] == 131072
            )
            results[name] = {
                k: data[k]
                for k in (
                    "completed",
                    "failed",
                    "total_input_tokens",
                    "total_output_tokens",
                    "duration",
                    "output_throughput",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                )
            }
            (run / "stable_summary.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(json.dumps({"round": name, **results[name]}), flush=True)
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            for _ in range(60):
                if server.poll() is not None:
                    break
                time.sleep(1)
            if server.poll() is None:
                raise RuntimeError(f"Owned server {server.pid} did not stop")


if __name__ == "__main__":
    main()
