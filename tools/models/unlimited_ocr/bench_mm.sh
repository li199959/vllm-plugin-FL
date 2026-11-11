#!/bin/bash
# Image-input load test for an Unlimited-OCR server (vllm bench serve, random-mm).
# Input size is set by image resolution; output length is fixed with --ignore-eos.
# Each scenario: one warmup round (64 requests, different seed), then the measured round.
# Usage: bench_mm.sh <out_dir> <port> [concurrency=64] [num_prompts=256]
set -u
OUT=$1; PORT=$2; CONC=${3:-64}; NUM=${4:-256}
TOK=${TOK:-/mnt/cpfs/xiangbin/codex/Unlimited-OCR}
mkdir -p "$OUT"; cd "$OUT"
# name  height  width  output_len
SCENARIOS="short 640 640 1024
mid 1754 1240 2048
long 3508 2480 4096"
run() {  # name h w out num seed file
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --host 127.0.0.1 --port "$PORT" --model baidu/Unlimited-OCR --tokenizer "$TOK" \
    --dataset-name random-mm --random-mm-bucket-config "{($2, $3, 1): 1.0}" \
    --random-mm-base-items-per-request 1 --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --random-input-len 16 --random-output-len "$4" --random-range-ratio 0 --ignore-eos \
    --extra-body '{"skip_special_tokens": false, "vllm_xargs": {"ngram_size": 35, "window_size": 128}}' \
    --num-prompts "$5" --max-concurrency "$CONC" --seed "$6" \
    --save-result --result-dir . --result-filename "$7.json" > "$7.log" 2>&1
}
echo "$SCENARIOS" | while read -r name h w out; do
  run "$name" "$h" "$w" "$out" 64 100 "${name}_warmup"
  run "$name" "$h" "$w" "$out" "$NUM" 1 "$name"
  python3 - "$name" <<'EOF'
import json, sys
n = sys.argv[1]; d = json.load(open(f"{n}.json"))
print(f"{n} ok {d['completed']}/{d['num_prompts']} in/req {d['total_input_tokens']/d['completed']:.0f} "
      f"out tok/s {d['output_throughput']:.1f} total tok/s {d['total_token_throughput']:.1f} req/s {d['request_throughput']:.3f} "
      f"TTFT mean {d['mean_ttft_ms']:.0f} p99 {d['p99_ttft_ms']:.0f} TPOT mean {d['mean_tpot_ms']:.2f} p99 {d['p99_tpot_ms']:.2f} "
      f"dur {d['duration']:.0f}s", flush=True)
EOF
done
echo DONE
