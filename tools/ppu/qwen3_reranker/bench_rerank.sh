#!/bin/bash
# Qwen3-Reranker-8B load test: the user's `vllm bench serve --backend vllm-rerank` command,
# L = 1024 / 4096 / 16384 / 32640, 64 concurrency, 256 requests, seed 0.
# Usage: bench_rerank.sh <out_dir> <port> [lengths="1024 4096 16384 32640"]
set -u
OUT=$1; PORT=$2; LENS=${3:-"1024 4096 16384 32640"}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); cd "$OUT"
for L in $LENS; do
  vllm bench serve --backend vllm-rerank --endpoint /rerank \
    --model Qwen3-Reranker-8B --tokenizer /mnt/cpfs/xiangbin/Qwen3-Reranker-8B/ \
    --host 127.0.0.1 --port "$PORT" \
    --dataset-name random-rerank --random-input-len "$L" --random-batch-size 1 --random-range-ratio 0 \
    --num-prompts 256 --max-concurrency 64 --request-rate inf --seed 0 \
    --ready-check-timeout-sec 0 --num-warmups 0 \
    --percentile-metrics e2el --metric-percentiles 50,95,99 \
    --request-id-prefix "rerank-$L-" \
    --save-result --save-detailed --result-dir "$OUT" --result-filename "qwen3_reranker_$L.json" \
    > "qwen3_reranker_$L.log" 2>&1
  python3 - "qwen3_reranker_$L" <<'EOF'
import json, sys
n = sys.argv[1]
try:
    d = json.load(open(f"{n}.json"))
except FileNotFoundError:
    print(f"{n} FAILED (no result json, see {n}.log)", flush=True); sys.exit(0)
print(f"{n} ok {d['completed']}/{d.get('num_prompts', d['completed'] + d.get('failed', 0))} "
      f"in/req {d['total_input_tokens'] / max(d['completed'], 1):.0f} "
      f"req/s {d['request_throughput']:.2f} tok/s {d['total_token_throughput']:.0f} "
      f"e2el mean {d['mean_e2el_ms']:.0f} p50 {d.get('p50_e2el_ms', 0):.0f} "
      f"p95 {d.get('p95_e2el_ms', 0):.0f} p99 {d.get('p99_e2el_ms', 0):.0f} ms dur {d['duration']:.1f}s",
      flush=True)
EOF
done
echo DONE
