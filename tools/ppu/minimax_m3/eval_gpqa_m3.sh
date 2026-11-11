#!/bin/bash
# GPQA Diamond accuracy for a running MiniMax-M3 server, following
# profiles/qwen36_35b_a3b_eval_20261007/REPORT.md (EvalScope 1.12, same local
# dataset, 0-shot, rule judge, batch 32, seed 42, full 198 questions, no cache).
# M3 is a thinking model: sampling follows its generation_config (T=1.0,
# top_p=0.95), thinking_mode=enabled, answer extracted after </mm:think>.
# Usage: [USE_CACHE=<previous work_dir>] eval_gpqa_m3.sh <port> <work_dir> [max_tokens=30000]
# USE_CACHE resumes an interrupted run: saved predictions are reused, only the rest is generated.
set -euo pipefail
PORT=$1; WORK=$2; MAXTOK=${3:-30000}
MODEL=${MODEL:-/mnt/cpfs/xiangbin/MiniMax-M3}
MID=MiniMax-M3-FL
EV=${EV:-/mnt/cpfs/xiangbin/codex/profiles/qwen27b_eval_gpqa_erqa_20261006}
LOCAL=$EV/datasets/datasets/AI-ModelScope--gpqa_diamond/snapshots/master
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 NUMEXPR_MAX_THREADS=16
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do sleep 15; done
$EV/evalenv/bin/evalscope eval \
  --model "$MODEL" \
  --model-id "$MID" \
  --eval-type openai_api \
  --eval-backend Native \
  --api-url "http://127.0.0.1:$PORT/v1" \
  --api-key EMPTY \
  --datasets gpqa_diamond \
  --dataset-args "{\"gpqa_diamond\": {\"local_path\": \"$LOCAL\", \"few_shot_num\": 0,
                   \"filters\": {\"remove_until\": \"</mm:think>\"}}}" \
  --generation-config "{
    \"timeout\": 3600.0, \"retries\": 2, \"retry_interval\": 10,
    \"batch_size\": 32, \"stream\": false, \"max_tokens\": $MAXTOK,
    \"temperature\": 1.0, \"top_p\": 0.95,
    \"presence_penalty\": 0.0, \"repetition_penalty\": 1.0, \"seed\": 42,
    \"extra_body\": {\"chat_template_kwargs\": {\"thinking_mode\": \"enabled\"}}
  }" \
  --eval-batch-size 32 \
  --repeats 1 \
  --seed 42 \
  --judge '{"strategy": "rule"}' \
  --no-timestamp \
  --enable-progress-tracker \
  ${USE_CACHE:+--use-cache "$USE_CACHE"} \
  --work-dir "$WORK"
python3 - "$WORK" "$MID" <<'EOF'
import json, sys
work, mid = sys.argv[1:]
r = json.load(open(f"{work}/reports/{mid}/gpqa_diamond.json"))
m = next(x for x in r["metrics"] if x["identity"]["name"] == "accuracy")
print(f"RESULT gpqa_diamond n={m['num']} accuracy={m['score'] * 100:.2f}%")
EOF
