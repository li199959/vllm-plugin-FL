#!/bin/bash
# Warm, mid-run decode profile of one M3 configuration.
# Usage: run_warm_profile.sh <hybrid|baseline> <out_dir> [input_len=1024]
# Starts the server (profiler armed, memory DEBUG logs), warms the input
# length, runs the 256-prompt benchmark and profiles 200 engine steps after
# 60 s (steady 64-way decode). Leaves the server running; stop it separately.
set -u
variant=$1; out=$2; il=${3:-1024}
mkdir -p "$out"
root=$(cd "$(dirname "$0")/../../.." && pwd)
# Optional: a vLLM logging config that enables DEBUG for vllm_fl.worker, to
# record the CUDA-graph memory estimate (vllm_fl.* loggers have no handler).
[ -n "${M3_LOGGING_CONFIG:-}" ] && export VLLM_LOGGING_CONFIG_PATH=$M3_LOGGING_CONFIG
export VLLM_FL_TOPK_DISPATCH=1
if [ "$variant" = baseline ]; then
  cd /mnt/cpfs/xiangbin/plugin_env_0906/vllm-plugin-FL
  export PYTHONPATH=$PWD VLLM_FL_MOE=deepgemm
  export VLLM_FL_FLAGOS_BLACKLIST=index_copy_,index,linear,addmm,mm,empty.memory_format
else
  cd "$root"
  export PYTHONPATH=$PWD VLLM_FL_MOE=hybrid VLLM_FL_M3_DENSE=hybrid VLLM_FL_M3_GEMMA_NORM=flaggems
  export VLLM_FL_M3_FUSED_IMPL=triton
  # Keep aten::index native: Gems index adds ~44 MiB per captured graph (~1 GiB KV).
  export VLLM_FL_FLAGOS_BLACKLIST=index_copy_,index,linear,addmm,mm,empty.memory_format
fi
prof=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,"torch_profiler_use_gzip":true,"ignore_frontend":true,"max_iterations":200}' "$out/trace")
setsid nohup vllm serve /mnt/cpfs/xiangbin/MiniMax-M3/ --host 0.0.0.0 --port 8100 \
  --tensor-parallel-size 16 --gpu-memory-utilization 0.85 --max-model-len 32768 \
  --block-size 128 --no-enable-prefix-caching --trust-remote-code \
  --profiler-config "$prof" > "$out/server.log" 2>&1 < /dev/null &
until curl -sf -o /dev/null http://127.0.0.1:8100/health; do
  grep -q 'umd error\|NCCL error\|Traceback' "$out/server.log" && { echo "!! startup failed"; exit 1; }
  sleep 15
done
echo "healthy $(date +%T)"
bench=(vllm bench serve --backend vllm --model /mnt/cpfs/xiangbin/MiniMax-M3/ --endpoint /v1/completions
       --host localhost --port 8100 --dataset-name random --ignore-eos --random-output-len 1024
       --max-concurrency 64 --random-input-len "$il")
cd "$out"
"${bench[@]}" --num-prompts 64 > warm.log 2>&1
"${bench[@]}" --num-prompts 256 --save-result --save-detailed --result-dir . --result-filename bench.json > bench.log 2>&1 &
bp=$!
sleep 60
curl -s -X POST http://127.0.0.1:8100/start_profile > /dev/null
echo "profiling from $(date +%T)"
wait $bp
curl -s -X POST http://127.0.0.1:8100/stop_profile > /dev/null
grep -E 'Output token throughput|Total token throughput|Mean TTFT|Mean TPOT' bench.log
grep -m1 -oE 'PIECEWISE CUDA graph memory: [^\n]*per-graph' server.log
grep -m1 -oE 'GPU KV cache size: [0-9,]+' server.log
