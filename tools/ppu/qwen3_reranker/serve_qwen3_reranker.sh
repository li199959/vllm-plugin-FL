#!/bin/bash
# Start Qwen3-Reranker-8B with the user's reference command (pooling runner,
# Qwen3ForSequenceClassification via hf_overrides, yes/no logit score, reranker
# chat template) and record argv/env for the run.
# Usage: [GPU_UTIL=0.9] [EXTRA_ARGS=...] serve_qwen3_reranker.sh <run_dir> <gpu> <port> <native|worktree> [profile=0|1]
#   native   : installed plugin, caller env unchanged (the user's reference command)
#   worktree : this checkout via PYTHONPATH; VLLM_FL_* switches come from the caller's env
set -eu
R=$1; GPUS=$2; PORT=$3; VARIANT=$4; PROFILE=${5:-0}
MODEL=${MODEL:-/mnt/cpfs/xiangbin/Qwen3-Reranker-8B/}
TEMPLATE=/opt/vllm/examples/pooling/score/template/qwen3_reranker.jinja
SKILL=/root/.codex/skills/vllm-gpu-coverage/scripts
root=$(cd "$(dirname "$0")/../../.." && pwd)
[ -e "$R" ] && { echo "run dir exists: $R"; exit 1; }
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 1; }
mkdir -p "$R"
R=$(cd "$R" && pwd)

export CUDA_VISIBLE_DEVICES=$GPUS
pypath=""
[ "$VARIANT" = worktree ] && pypath=$root
args=(vllm serve "$MODEL" --served-model-name Qwen3-Reranker-8B --runner pooling --port "$PORT"
      --max-model-len "${MAX_LEN:-32768}" --gpu-memory-utilization "${GPU_UTIL:-0.9}"
      --no-enable-prefix-caching
      --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
      --chat-template "$TEMPLATE")
[ -n "${EXTRA_ARGS:-}" ] && args+=(${EXTRA_ARGS})
if [ "$PROFILE" = 1 ]; then
  export FLAGGEMS_ENABLE_OPLIST_PATH=$R/flaggems.txt
  pypath=$SKILL${pypath:+:$pypath}
  prof=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"torch_profiler_with_memory":false,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":false,"ignore_frontend":false,"delay_iterations":0,"max_iterations":0,"warmup_iterations":0,"wait_iterations":0}' "$R")
  args+=(--worker-extension-cls coverage_worker_audit.CoverageWorkerAudit --profiler-config "$prof")
fi
[ -n "$pypath" ] && export PYTHONPATH=$pypath

python3 - "$R" "$VARIANT" "${args[@]}" <<'EOF'
import json, os, sys
run, variant, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
env = {k: v for k, v in os.environ.items()
       if k.startswith(("VLLM_", "FLAGGEMS_", "CUDA_VISIBLE", "PYTHONPATH", "TRITON_"))}
json.dump({"variant": variant, "argv": argv, "env": env}, open(f"{run}/serve.json", "w"), indent=1)
EOF
cd /tmp
setsid nohup "${args[@]}" > "$R/server.log" 2>&1 < /dev/null &
echo "$!" > "$R/server.pid"
echo "started pid $(cat "$R/server.pid") on GPU $GPUS port $PORT -> $R"
