#!/bin/bash
# GPQA Diamond / ERQA accuracy for a running Qwen3-VL server, following
# profiles/qwen36_35b_a3b_eval_20261007/REPORT.md (EvalScope 1.12, same local
# datasets, rule judge, batch 32, seed 42, full set, no cache).
# Qwen3-VL-30B-A3B-Instruct is a non-thinking model, so sampling follows its
# generation_config (T=0.7, top_p=0.8, top_k=20) and thinking is not enabled.
# Usage: eval_qwen3_vl.sh <gpqa_diamond|erqa> <port> <work_dir> [model_id]
#   env: PRESENCE (presence_penalty, default 0.0; Qwen recommends 1.5 for Instruct),
#        MAXTOK (default 16384), REPEATS (default 1), TEMP (default 0.7)
set -euo pipefail
DS=$1; PORT=$2; WORK=$3; MID=${4:-Qwen3-VL-30B-A3B-FL}
PRESENCE=${PRESENCE:-0.0}; MAXTOK=${MAXTOK:-16384}; REPEATS=${REPEATS:-1}; TEMP=${TEMP:-0.7}
EV=${EV:-/mnt/cpfs/xiangbin/codex/profiles/qwen27b_eval_gpqa_erqa_20261006}
case $DS in
  gpqa_diamond) LOCAL=$EV/datasets/datasets/AI-ModelScope--gpqa_diamond/snapshots/master
                DSARGS="{\"gpqa_diamond\": {\"local_path\": \"$LOCAL\", \"few_shot_num\": 0}}" ;;
  erqa)         LOCAL=$EV/datasets/datasets/evalscope--ERQA/snapshots/master
                DSARGS="{\"erqa\": {\"local_path\": \"$LOCAL\"}}" ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 NUMEXPR_MAX_THREADS=16
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do sleep 15; done
$EV/evalenv/bin/evalscope eval \
  --model qwen3-vl \
  --model-id "$MID" \
  --eval-type openai_api \
  --eval-backend Native \
  --api-url "http://127.0.0.1:$PORT/v1" \
  --api-key EMPTY \
  --datasets "$DS" \
  --dataset-args "$DSARGS" \
  --generation-config "{
    \"timeout\": 3600.0, \"retries\": 2, \"retry_interval\": 10,
    \"batch_size\": 32, \"stream\": false, \"max_tokens\": $MAXTOK,
    \"temperature\": $TEMP, \"top_p\": 0.8, \"top_k\": 20,
    \"presence_penalty\": $PRESENCE, \"repetition_penalty\": 1.0, \"seed\": 42,
    \"extra_body\": {\"min_p\": 0.0}
  }" \
  --eval-batch-size 32 \
  --repeats $REPEATS \
  --seed 42 \
  --judge '{"strategy": "rule"}' \
  --no-timestamp \
  --enable-progress-tracker \
  --work-dir "$WORK"
python3 - "$WORK" "$MID" "$DS" <<'EOF'
import json, sys
work, mid, ds = sys.argv[1:]
r = json.load(open(f"{work}/reports/{mid}/{ds}.json"))
m = next(x for x in r["metrics"] if x["identity"]["name"] == "accuracy")
print(f"RESULT {ds} n={m['num']} accuracy={m['score'] * 100:.2f}%")
EOF
