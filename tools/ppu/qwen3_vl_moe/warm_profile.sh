#!/bin/bash
# Warm, mid-run profile of one Qwen3-VL configuration (no prefix cache):
# start the server with the torch profiler armed (no stacks/shapes, 300 engine
# steps), warm up with one full round, then profile 300 steps of a second round
# 40 s in (steady mixed prefill/decode). Prints the per-kernel GPU time table.
# Usage: [VLLM_FL_* env] [INPUT_LEN=4096] warm_profile.sh <run_dir> <gpus> <port>
set -u
R=$1; GPUS=$2; PORT=$3; IL=${INPUT_LEN:-4096}
T=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$R" && R=$(cd "$R" && pwd)
prof=$(printf '{"profiler":"torch","torch_profiler_dir":"%s/trace","torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,"torch_profiler_use_gzip":true,"ignore_frontend":true,"max_iterations":300}' "$R")
EXTRA_ARGS="--no-enable-prefix-caching --profiler-config $prof" \
  "$T/serve_qwen3_vl.sh" "$R/serve" "$GPUS" "$PORT" worktree 0 || exit 1
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
  grep -q 'Traceback\|umd error' "$R/serve/server.log" && { echo "!! startup error"; exit 1; }
  sleep 15
done
bench=(vllm bench serve --backend vllm --model qwen3-vl --tokenizer /mnt/cpfs/xiangbin/Qwen3-VL-30B-A3B-Instruct/
       --endpoint /v1/completions --host 127.0.0.1 --port "$PORT" --dataset-name random
       --random-input-len "$IL" --random-output-len 1024 --ignore-eos --max-concurrency 64 --num-prompts 128)
cd "$R"
"${bench[@]}" > warm.log 2>&1
"${bench[@]}" --save-result --result-dir . --result-filename bench.json > bench.log 2>&1 &
bp=$!
sleep 40
curl -s -X POST "http://127.0.0.1:$PORT/start_profile" > /dev/null
echo "profiling from $(date +%T)"
wait $bp
curl -s -X POST "http://127.0.0.1:$PORT/stop_profile" > /dev/null
grep -E 'Output token throughput|Mean TTFT|Mean TPOT' bench.log
sleep 60
python3 - "$(ls -S "$R"/trace/*rank0*.json.gz 2>/dev/null | head -1)" <<'EOF'
import collections, gzip, sys
import ijson
tot = collections.defaultdict(lambda: [0, 0.0])
lo, hi = None, 0
with gzip.open(sys.argv[1], "rb") as f:
    for ev in ijson.items(f, "traceEvents.item", use_float=True):
        if ev.get("cat") == "kernel":
            r = tot[ev["name"]]; r[0] += 1; r[1] += ev.get("dur", 0.0)
            lo = ev["ts"] if lo is None else min(lo, ev["ts"]); hi = max(hi, ev["ts"] + ev.get("dur", 0))
busy = sum(t for _, t in tot.values())
print(f"window {(hi - lo) / 1e6:.2f}s  kernel busy {busy / 1e6:.2f}s ({100 * busy / (hi - lo):.0f}%)")
for n, (c, t) in sorted(tot.items(), key=lambda x: -x[1][1])[:30]:
    print(f"{t / 1e3:9.1f}ms {100 * t / busy:5.1f}% {c:7d} {t / c:8.1f}us  {n[:90]}")
EOF
