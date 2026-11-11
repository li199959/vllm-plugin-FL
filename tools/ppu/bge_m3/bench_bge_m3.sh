#!/bin/bash
# Embedding throughput for a bge-m3 server: one warmup + N rounds of random prompts,
# a different --seed per round so nothing is reused between rounds.
# Usage: [INPUT_LEN=512] [CONC=64] [NUM=2048] bench_bge_m3.sh <run_dir> <port> [rounds=3]
# Prints per round: req/s, input tok/s, mean/p99 E2E latency, and Triton JIT compiles.
set -u
R=$(cd "$1" && pwd); PORT=$2; ROUNDS=${3:-3}
IL=${INPUT_LEN:-512}; CONC=${CONC:-64}; NUM=${NUM:-2048}
MODEL=${MODEL:-/mnt/cpfs/xiangbin/codex/bge-m3}
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
  [ -f "$R/server.log" ] && grep -q 'Traceback\|umd error' "$R/server.log" && {
    echo "!! startup error"; exit 1; }
  sleep 10
done
echo "healthy $(date +%T) IL=$IL conc=$CONC num=$NUM"
mkdir -p "$R/bench_il$IL"; cd "$R/bench_il$IL"
for i in warmup $(seq "$ROUNDS"); do
  seed=$((1000 + ${i/warmup/0})); touch .t0
  vllm bench serve --backend openai-embeddings --model bge-m3 --tokenizer "$MODEL" \
    --endpoint /v1/embeddings --host 127.0.0.1 --port "$PORT" --dataset-name random \
    --random-input-len "$IL" --random-output-len 1 --random-range-ratio 0 --seed "$seed" \
    --max-concurrency "$CONC" --num-prompts "$NUM" --percentile-metrics e2el \
    --metric-percentiles 50,99 --save-result --result-dir . --result-filename "round_$i.json" \
    > "round_$i.log" 2>&1
  # Triton compiles + Gems generated sources (Gems codegens index/pointwise per shape).
  jit=$(find /root/.triton/cache /root/.flaggems/code_cache -newer .t0 -type f \
        \( -name '*.json' -o -name '*.py' \) 2>/dev/null | wc -l)
  python3 - "$i" "$jit" <<'PY'
import json, sys
i, jit = sys.argv[1], sys.argv[2]
d = json.load(open(f"round_{i}.json"))
print(f"round {i} {d['completed']}/{d['num_prompts']} req/s {d['request_throughput']:.1f} "
      f"in tok/s {d['total_token_throughput']:.0f} e2el mean {d['mean_e2el_ms']:.1f} "
      f"p99 {d['p99_e2el_ms']:.1f} ms jit {jit}", flush=True)
PY
done
echo DONE
