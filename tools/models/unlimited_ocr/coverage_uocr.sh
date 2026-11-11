#!/bin/bash
# Coverage run for a profiled Unlimited-OCR server started with
# `serve_uocr.sh <run_dir> <gpu> <port> 1` (vllm-gpu-coverage skill):
# two warmup rounds + one profiled round over the page set, then scan,
# provenance, coverage.
# Usage: [CONC=16] coverage_uocr.sh <run_dir> <port> <page_dir>
set -u
R=$(cd "$1" && pwd); PORT=$2; PAGES=$3; CONC=${CONC:-16}
S=/root/.codex/skills/vllm-gpu-coverage/scripts
here=$(cd "$(dirname "$0")" && pwd)
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
  grep -q 'EngineCore failed' "$R/server.log" && { echo "!! startup error"; exit 1; }
  sleep 10
done
echo "healthy $(date +%T)"
python3 "$here/bench_ocr.py" --port "$PORT" --concurrency "$CONC" --rounds 3 --profile \
  --out "$R/bench.json" "$PAGES"/page_*.png || exit 1
cp "$R/flaggems.txt" "$R/flaggems_final.txt" 2>/dev/null
sleep 30  # let the worker trace finish writing
trace=$(ls -t "$R"/rank0.*.pt.trace.json.gz "$R"/dp0_pp0_tp0_*.pt.trace.json.gz 2>/dev/null | head -1)
python3 $S/scan_trace.py "$trace" "$R/rank0.coverage_scan.json" --require-graph > "$R/scan_rank0.log" 2>&1
tail -n1 "$R/scan_rank0.log"
python3 $S/provenance.py --scans "$R"/rank0.coverage_scan.json --cache /root/.cache/vllm/torch_compile_cache \
  --cache /root/.triton/cache --cache /root/.flaggems --cache /tmp/torchinductor_root \
  --out "$R/provenance.json" 2>&1 | tail -n1
python3 $S/coverage.py --scans "$R"/rank0.coverage_scan.json --audit "$R"/worker_audit_rank*.json \
  --log "$R/flaggems.txt" --provenance "$R/provenance.json" --model Unlimited-OCR \
  --out "$R/gpu_coverage" 2>&1 | tail -n1
python3 - "$R" <<'EOF'
import csv, json, sys
r = sys.argv[1]
print("verification", json.load(open(f"{r}/gpu_coverage/verification.json"))["checks_passed"])
print("combined", json.load(open(f"{r}/gpu_coverage/summary.json"))["combined"])
rows = list(csv.DictReader(open(f"{r}/gpu_coverage/coverage.csv", encoding="utf-8-sig")))
print("uncovered:", [x["operator_name"][:70] for x in rows if x["covered"] != "true"])
EOF
