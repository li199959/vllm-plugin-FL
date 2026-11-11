#!/bin/bash
# The user's EvalScope embedding perf cases against a bge-m3 server on <port>:
# random_embedding, 1024 / 4096 / 8096 tokens, 64 parallel, 256 requests, 10 warmup.
# Server must be started with NAME=BAAI/bge-m3 (the --model below).
# Usage: evalscope_perf_bge_m3.sh <out_dir> <port> [lengths="1024 4096 8096"]
set -u
OUT=$1; PORT=$2; LENS=${3:-"1024 4096 8096"}
EV=${EV:-/mnt/cpfs/xiangbin/codex/profiles/qwen27b_eval_gpqa_erqa_20261006/evalenv/bin/evalscope}
TOK=${TOK:-/mnt/cpfs/xiangbin/codex/bge-m3}  # local copy of BAAI/bge-m3
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || { echo "server on $PORT not healthy"; exit 1; }
cd "$OUT"
for L in $LENS; do
  "$EV" perf \
    --model BAAI/bge-m3 \
    --url "http://127.0.0.1:$PORT/v1/embeddings" \
    --api openai_embedding \
    --dataset random_embedding \
    --min-prompt-length "$L" \
    --max-prompt-length "$L" \
    --tokenizer-path "$TOK" \
    --parallel 64 \
    --number 256 \
    --warmup-num 10 \
    --outputs-dir "$OUT/len$L" > "$OUT/len$L.log" 2>&1
  echo "len $L exit $?"
done
