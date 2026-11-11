#!/bin/bash
# Coverage run for a profiled Qwen3-VL server started with
# `serve_qwen3_vl.sh <run_dir> <gpus> <port> <variant> 1` (vllm-gpu-coverage skill):
# two warmups + one profiled round of 4k/1k, then scan, provenance, coverage.
# Usage: coverage_qwen3_vl.sh <run_dir> <port>
set -u
R=$(cd "$1" && pwd); PORT=$2
S=/root/.codex/skills/vllm-gpu-coverage/scripts
[ -f "$R/bench.json" ] || { echo "missing $R/bench.json (bench argv for port $PORT)"; exit 1; }
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
  grep -q 'Traceback\|umd error\|NCCL error' "$R/server.log" && { echo "!! startup error"; exit 1; }
  sleep 15
done
echo "healthy $(date +%T)"
cd /tmp
python3 $S/run_benchmark.py --command-json "$R/bench.json" --out "$R/bench" \
  --base-url "http://127.0.0.1:$PORT" --flaggems-log "$R/flaggems.txt" --expected-prompts 128 \
  --expected-input-tokens 524288 --expected-output-tokens 131072 2>&1 | grep -v 'still running'
sleep 60  # let the worker traces finish writing
for r in 0 1; do
  python3 $S/scan_trace.py "$(ls "$R"/dp0_pp0_tp${r}_*.pt.trace.json.gz)" "$R/rank$r.coverage_scan.json" \
    --require-graph > "$R/scan_rank$r.log" 2>&1 &
done
wait
tail -qn1 "$R"/scan_rank*.log
python3 $S/provenance.py --scans "$R"/rank*.coverage_scan.json --cache /root/.cache/vllm/torch_compile_cache \
  --cache /root/.triton/cache --cache /root/.flaggems --out "$R/provenance.json" 2>&1 | tail -n1
python3 $S/coverage.py --scans "$R"/rank*.coverage_scan.json --audit "$R"/worker_audit_rank*.json \
  --log "$R/flaggems.txt" --provenance "$R/provenance.json" --model Qwen3-VL-30B-A3B-Instruct \
  --out "$R/gpu_coverage" 2>&1 | tail -n1
python3 - "$R" <<'EOF'
import csv, json, sys
r = sys.argv[1]
print("verification", json.load(open(f"{r}/gpu_coverage/verification.json"))["checks_passed"])
rows = list(csv.DictReader(open(f"{r}/gpu_coverage/coverage.csv", encoding="utf-8-sig")))
print("uncovered:", [x["operator_name"][:60] for x in rows if x["covered"] != "true"])
for k in ("warmup1", "warmup2", "profile"):
    d = json.load(open(f"{r}/bench/{k}.json"))
    print(k, f"out {d['output_throughput']:.0f}", f"TPOT {d['mean_tpot_ms']:.2f}")
EOF
