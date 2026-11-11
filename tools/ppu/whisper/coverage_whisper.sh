#!/bin/bash
# Coverage run for a profiled whisper server started with
# `serve_whisper.sh <run_dir> <gpu> <port> <variant> 1` (vllm-gpu-coverage skill):
# two warmups + one profiled round of the same LibriSpeech subset, then scan,
# provenance, coverage. Encoder and prefill steps run outside CUDA graphs
# (enc-dec skip_compiled), so graph replay is not required.
# Usage: [NUM=256] [DATA=...] coverage_whisper.sh <run_dir> <port>
set -u
R=$(cd "$1" && pwd); PORT=$2
NUM=${NUM:-256}
DATA=${DATA:-/mnt/cpfs/xiangbin/codex/datasets/librispeech/LibriSpeech/test-clean}
S=/root/.codex/skills/vllm-gpu-coverage/scripts
B=$(cd "$(dirname "$0")" && pwd)/bench_asr.py
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
  grep -q 'Traceback\|umd error' "$R/server.log" && { echo "!! startup error"; exit 1; }
  sleep 10
done
echo "healthy $(date +%T)"
mkdir -p "$R/bench"; cd "$R/bench"
for label in warmup1 warmup2 profile; do
  extra=(); [ $label = profile ] && { cp "$R/flaggems.txt" flaggems_after_warmup.txt; extra=(--profile); }
  python3 "$B" --port "$PORT" --data "$DATA" --num "$NUM" --concurrency 64 --warmup 0 \
    --out "$label.json" "${extra[@]}" > "$label.log" 2>&1 || {
      echo "!! $label failed"; curl -s -X POST "http://127.0.0.1:$PORT/stop_profile"; exit 1; }
  python3 - "$label" "$NUM" <<'EOF' || exit 1
import json, sys
label, num = sys.argv[1], int(sys.argv[2])
s = json.load(open(f"{label}.json"))["summary"]
assert s["failed"] == 0 and s["requests"] == num, (label, s)
print(label, s["requests"], "req", s["req_per_s"], "req/s", "WER", s["wer"], flush=True)
EOF
done
cp "$R/flaggems.txt" flaggems.txt
sleep 60  # let the worker trace finish writing
python3 $S/scan_trace.py "$(ls "$R"/dp0_pp0_tp0_*.pt.trace.json.gz "$R"/rank0.*.pt.trace.json.gz 2>/dev/null | head -1)" \
  "$R/rank0.coverage_scan.json" > "$R/scan_rank0.log" 2>&1
tail -n1 "$R/scan_rank0.log"
python3 $S/provenance.py --scans "$R"/rank0.coverage_scan.json --cache /root/.cache/vllm/torch_compile_cache \
  --cache /root/.triton/cache --cache /root/.flaggems --cache /tmp/torchinductor_root --out "$R/provenance.json" 2>&1 | tail -n1
python3 $S/coverage.py --scans "$R"/rank0.coverage_scan.json --audit "$R"/worker_audit_rank*.json \
  --log "$R/flaggems.txt" --provenance "$R/provenance.json" --model whisper-large-v3 \
  --allow-no-graph --out "$R/gpu_coverage" 2>&1 | tail -n1
python3 - "$R" <<'EOF'
import csv, json, sys
r = sys.argv[1]
print("verification", json.load(open(f"{r}/gpu_coverage/verification.json"))["checks_passed"])
s = json.load(open(f"{r}/gpu_coverage/summary.json"))["combined"]
print("combined", s)
rows = list(csv.DictReader(open(f"{r}/gpu_coverage/coverage.csv", encoding="utf-8-sig")))
print("uncovered:", [x["operator_name"][:70] for x in rows if x["covered"] != "true"])
EOF
