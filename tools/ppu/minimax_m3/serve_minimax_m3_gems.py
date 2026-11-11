"""Launch and record an explicit MiniMax-M3 backend comparison."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--tensor-parallel-size", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--log-file", action="store_true")
    parser.add_argument(
        "--moe", choices=("deepgemm", "flaggems", "hybrid"), default="deepgemm"
    )
    parser.add_argument(
        "--dense", choices=("native", "flaggems", "hybrid"), default="native"
    )
    parser.add_argument("--norm", choices=("native", "flaggems"), default="native")
    parser.add_argument("--fused", choices=("native", "triton"), default="native")
    parser.add_argument("--index", choices=("native", "flaggems"), default="native")
    parser.add_argument(
        "--coverage-skill-dir",
        type=Path,
        default=Path("/root/.codex/skills/vllm-gpu-coverage"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    selected = {
        "VLLM_FL_TOPK_DISPATCH": "1",
        "VLLM_FL_MOE": args.moe,
        "VLLM_FL_FLAGOS_BLACKLIST": (
            "index_copy_,index,linear,addmm,mm,empty.memory_format"
            if args.dense != "flaggems"
            else "index_copy_,index,empty.memory_format"
        ),
        "VLLM_FL_M3_DENSE": args.dense,
        "VLLM_FL_M3_GEMMA_NORM": args.norm,
        "VLLM_FL_M3_FUSED_IMPL": args.fused,
        "FLAGGEMS_ENABLE_OPLIST_PATH": str(run / "flaggems.txt"),
        "VLLM_CACHE_ROOT": str(run / "cache" / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(run / "cache" / "inductor"),
    }
    if args.index == "flaggems":
        # Each worker owns a cache. Newly generated index specializations must
        # not contend for a shared SQLite writer in the request path.
        selected["FLAGGEMS_DB_URL"] = "sqlite:///:memory:"
    if args.index == "flaggems":
        selected["VLLM_FL_FLAGOS_BLACKLIST"] = ",".join(
            op
            for op in selected["VLLM_FL_FLAGOS_BLACKLIST"].split(",")
            if op != "index"
        )
    paths = [str(root)]
    if args.profile:
        paths.append(str(args.coverage_skill_dir / "scripts"))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    selected["PYTHONPATH"] = os.pathsep.join(paths)
    env.update(selected)
    command = [
        "vllm",
        "serve",
        str(Path(args.model).resolve()),
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "32768",
        "--block-size",
        "128",
        "--trust-remote-code",
    ]
    if args.profile:
        config = {
            "profiler": "torch",
            "torch_profiler_dir": str(run),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": True,
            "torch_profiler_dump_cuda_time_total": False,
            "ignore_frontend": False,
            "delay_iterations": 0,
            "max_iterations": 0,
            "warmup_iterations": 0,
            "wait_iterations": 0,
        }
        command += [
            "--worker-extension-cls",
            "coverage_worker_audit.CoverageWorkerAudit",
            "--profiler-config",
            json.dumps(config),
        ]
    bench = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        str(Path(args.model).resolve()),
        "--endpoint",
        "/v1/completions",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "4096",
        "--random-output-len",
        "1024",
        "--max-concurrency",
        "64",
        "--num-prompts",
        "128",
        "--ignore-eos",
    ]
    sources = [
        root / "vllm_fl" / p
        for p in [
            "ops/minimax_m3_triton.py",
            "ops/custom_ops.py",
            "ops/fused_moe/fused_moe_utils.py",
        ]
    ] + sorted((root / "vllm_fl/optimizations").rglob("*.py"))
    for name, data in [
        ("serve.argv.json", command),
        ("serve.env.json", selected),
        ("bench.json", bench),
        (
            "source_hashes.json",
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        ),
    ]:
        (run / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (run / "server.pid").write_text(str(os.getpid()) + "\n")
    print(f"Starting MiniMax-M3; run artifacts: {run}", flush=True)
    if args.log_file:
        sys.stdout.flush()
        sys.stderr.flush()
        with (run / "server.log").open("w") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
