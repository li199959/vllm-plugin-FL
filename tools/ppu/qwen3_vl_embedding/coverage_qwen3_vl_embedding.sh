#!/bin/bash
# Coverage run for a profiled Qwen3-VL-Embedding server started with
# `serve_qwen3_vl_embedding.sh <run_dir> <gpu> <port> <variant> 1` (vllm-gpu-coverage skill):
# two warmups + one profiled round of the same text case, then scan, provenance, coverage.
# Same driver as tools/ppu/bge_m3/coverage_bge_m3.sh (the skill's run_benchmark.py
# expects a `failed` field that vLLM's embedding result JSON does not have).
# Usage: [INPUT_LEN=1024] [NUM=256] coverage_qwen3_vl_embedding.sh <run_dir> <port>
set -u
R=$(cd "$1" && pwd); PORT=$2
IL=${INPUT_LEN:-1024}; NUM=${NUM:-256}
S=/root/.codex/skills/vllm-gpu-coverage/scripts
MODEL=${MODEL:-/mnt/cpfs/xiangbin/Qwen3-VL-Embedding-8B}
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
  grep -q 'Traceback\|umd error' "$R/server.log" && { echo "!! startup error"; exit 1; }
  sleep 10
done
echo "healthy $(date +%T)"
mkdir -p "$R/bench"; cd "$R/bench"
for label in warmup1 warmup2 profile; do
  extra=(); [ $label = profile ] && { cp "$R/flaggems.txt" flaggems_after_warmup.txt; extra=(--profile); }
  vllm bench serve --backend openai-embeddings --model qwen3-vl-embedding --tokenizer "$MODEL" \
    --endpoint /v1/embeddings --host 127.0.0.1 --port "$PORT" --dataset-name random \
    --random-input-len "$IL" --random-output-len 1 --random-range-ratio 0 --seed 1000 \
    --max-concurrency 64 --num-prompts "$NUM" \
    --extra-body '{"add_special_tokens":true,"encoding_format":"float"}' \
    --save-result --result-dir . --result-filename "$label.json" "${extra[@]}" > "$label.log" 2>&1 || {
      echo "!! $label failed"; curl -s -X POST "http://127.0.0.1:$PORT/stop_profile"; exit 1; }
  python3 - "$label" "$NUM" <<'EOF' || exit 1
import json, sys
label, num = sys.argv[1], int(sys.argv[2])
d = json.load(open(f"{label}.json"))
assert d["completed"] == num, (label, d["completed"], num)
print(label, d["completed"], "req", f"{d['request_throughput']:.2f} req/s", flush=True)
EOF
done
cp "$R/flaggems.txt" flaggems.txt
sleep 60  # let the worker trace finish writing
python3 $S/scan_trace.py "$(ls "$R"/dp0_pp0_tp0_*.pt.trace.json.gz "$R"/rank0.*.pt.trace.json.gz 2>/dev/null | head -1)" "$R/rank0.coverage_scan.json" \
  --require-graph > "$R/scan_rank0.log" 2>&1
tail -n1 "$R/scan_rank0.log"
python3 $S/provenance.py --scans "$R"/rank0.coverage_scan.json --cache /root/.cache/vllm/torch_compile_cache \
  --cache /root/.triton/cache --cache /root/.flaggems --cache /tmp/torchinductor_root --out "$R/provenance.json" 2>&1 | tail -n1
python3 $S/coverage.py --scans "$R"/rank0.coverage_scan.json --audit "$R"/worker_audit_rank*.json \
  --log "$R/flaggems.txt" --provenance "$R/provenance.json" --model qwen3-vl-embedding \
  --out "$R/gpu_coverage" 2>&1 | tail -n1
python3 - "$R" <<'EOF'
import csv, json, sys
r = sys.argv[1]
print("verification", json.load(open(f"{r}/gpu_coverage/verification.json"))["checks_passed"])
s = json.load(open(f"{r}/gpu_coverage/summary.json"))["combined"]
print("combined", s)
rows = list(csv.DictReader(open(f"{r}/gpu_coverage/coverage.csv", encoding="utf-8-sig")))
print("uncovered:", [x["operator_name"][:70] for x in rows if x["covered"] != "true"])
EOF
