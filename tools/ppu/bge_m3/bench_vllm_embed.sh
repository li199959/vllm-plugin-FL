#!/bin/bash
# bge-m3 embedding load test with `vllm bench serve` (user's command), lengths 1k/2k/4k/8k.
# Per length: one warmup round (seed 100, not reported) then the measured round (seed 0).
# Usage: bench_vllm_embed.sh <out_dir> <port> [lengths="1024 2048 4096 8192"]
# Note: --probe-request-rate from the original command is not supported by local vLLM 0.24.
set -u
OUT=$1; PORT=$2; LENS=${3:-"1024 2048 4096 8192"}
TOK=${TOK:-/mnt/cpfs/xiangbin/codex/bge-m3}
mkdir -p "$OUT"; cd "$OUT"
run() {  # len seed tag
  vllm bench serve --backend openai-embeddings --model bge-m3 --tokenizer "$TOK" \
    --endpoint /v1/embeddings --host 127.0.0.1 --port "$PORT" --dataset-name random \
    --random-input-len "$1" --random-output-len 1 --random-range-ratio 0 --random-prefix-len 0 \
    --random-batch-size 1 --max-concurrency 64 --num-prompts 256 --request-rate inf \
    --burstiness 1 --seed "$2" --ready-check-timeout-sec 0 --num-warmups 0 \
    --percentile-metrics e2el --metric-percentiles 50,95,99 \
    --extra-body '{"add_special_tokens":true,"truncate_prompt_tokens":null,"encoding_format":"float"}' \
    --request-id-prefix "bge-random-$3-" --save-result --save-detailed --result-dir . \
    --result-filename "$3.json" > "$3.log" 2>&1
}
for L in $LENS; do
  run "$L" 100 "len${L}_warmup"
  run "$L" 0 "len${L}"
  python3 - "len${L}" <<'EOF'
import json, sys
n = sys.argv[1]; d = json.load(open(f"{n}.json"))
p = {k: v for k, v in d.items() if k.startswith("p") and "e2el" in k}
print(f"{n} ok {d['completed']}/{len(d['input_lens'])} in/req {d['total_input_tokens']/max(d['completed'],1):.0f} "
      f"req/s {d['request_throughput']:.2f} tok/s {d['total_token_throughput']:.0f} "
      f"e2el mean {d['mean_e2el_ms']:.0f} p50 {d.get('p50_e2el_ms',0):.0f} p95 {d.get('p95_e2el_ms',0):.0f} "
      f"p99 {d.get('p99_e2el_ms',0):.0f} ms dur {d['duration']:.1f}s", flush=True)
EOF
done
echo DONE
