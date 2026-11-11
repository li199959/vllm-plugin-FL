#!/bin/bash
# Wait for a Qwen3-VL server, then run N rounds of the reference case
# (4k/1k, 64 concurrency, 128 requests, ignore-eos); one summary line per round
# including ITL stalls (>1 s) and Triton kernels compiled during the round.
# Usage: [INPUT_LEN=4096] bench_qwen3_vl.sh <run_dir> <port> [rounds=5]
set -u
R=$1; PORT=$2; N=${3:-5}
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
  grep -q 'Traceback\|umd error\|NCCL error' "$R/server.log" && { echo "!! startup error"; exit 1; }
  sleep 15
done
echo "healthy $(date +%T) $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$R/server.log")"
cd "$R"
for i in $(seq 0 $((N - 1))); do
  touch ".round$i.start"
  vllm bench serve --backend vllm --model qwen3-vl --tokenizer /mnt/cpfs/xiangbin/Qwen3-VL-30B-A3B-Instruct/ \
    --endpoint /v1/completions --host 127.0.0.1 --port "$PORT" --dataset-name random \
    --random-input-len "${INPUT_LEN:-4096}" --random-output-len 1024 --ignore-eos --max-concurrency 64 \
    --num-prompts 128 --save-result --save-detailed --result-dir . --result-filename "round$i.json" \
    > "round$i.log" 2>&1
  jit=$(find ~/.triton/cache -newer ".round$i.start" -name '*.json' ! -name '__grp__*' 2>/dev/null | wc -l)
  python3 - "$i" "$jit" <<'EOF'
import json, sys
i, jit = sys.argv[1:]
d = json.load(open(f"round{i}.json"))
itl = [x for r in d.get("itls", []) for x in r]
print(f"round {i} completed {d['completed']}/128 out {d['output_throughput']:.0f} tok/s "
      f"TTFT mean {d['mean_ttft_ms']:.0f} med {d['median_ttft_ms']:.0f} p99 {d['p99_ttft_ms']:.0f} "
      f"TPOT mean {d['mean_tpot_ms']:.2f} p99 {d['p99_tpot_ms']:.2f} "
      f"itl>1s {sum(x > 1 for x in itl)} jit {jit}")
EOF
done
