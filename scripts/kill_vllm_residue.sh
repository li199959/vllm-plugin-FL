#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
PID_FILE="${PID_FILE:-vllm-qwen3.6-35b-a3b.pid}"
MODEL_PATTERN="${MODEL_PATTERN:-Qwen3.6-35B-A3B|qwen3.6-35b-a3b}"
DRY_RUN=0
SKIP_KILL9=0
INCLUDE_GPU=0

usage() {
  cat <<'EOF'
Usage:
  scripts/kill_vllm_residue.sh [options]

Options:
  --port PORT             Port to clean, default: 8000
  --pid-file FILE         vLLM pid file, default: vllm-qwen3.6-35b-a3b.pid
  --pattern REGEX         Process command regex, default: Qwen3.6-35B-A3B|qwen3.6-35b-a3b
  --dry-run               Print matched processes without killing
  --no-kill9              Do not escalate to SIGKILL after SIGTERM
  --include-gpu           Also kill visible processes using /dev/nvidia* or listed by nvidia-smi
  -h, --help              Show this help

Environment overrides:
  PORT=8000 PID_FILE=... MODEL_PATTERN=...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --pid-file)
      PID_FILE="$2"
      shift 2
      ;;
    --pattern)
      MODEL_PATTERN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-kill9)
      SKIP_KILL9=1
      shift
      ;;
    --include-gpu)
      INCLUDE_GPU=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

declare -A PIDS=()
INVISIBLE_GPU_PIDS=()

add_pid() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ "$pid" == "$$" ]] && return 0
  kill -0 "$pid" 2>/dev/null || return 0
  PIDS["$pid"]=1
}

add_descendants() {
  local pid="$1"
  local children
  children="$(pgrep -P "$pid" 2>/dev/null || true)"
  local child
  for child in $children; do
    add_pid "$child"
    add_descendants "$child"
  done
}

if [[ -f "$PID_FILE" ]]; then
  add_pid "$(tr -cd '0-9' < "$PID_FILE")"
fi

if command -v lsof >/dev/null 2>&1; then
  while read -r pid; do
    add_pid "$pid"
  done < <(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
elif command -v fuser >/dev/null 2>&1; then
  while read -r pid; do
    add_pid "$pid"
  done < <(fuser -n tcp "$PORT" 2>/dev/null | tr ' ' '\n' || true)
else
  while read -r pid; do
    add_pid "$pid"
  done < <(ss -lptn "sport = :$PORT" 2>/dev/null \
    | sed -nE 's/.*pid=([0-9]+).*/\1/p' || true)
fi

while read -r pid _cmd; do
  add_pid "$pid"
done < <(ps -eo pid=,args= | grep -E "$MODEL_PATTERN" | grep -v grep || true)

if [[ "$INCLUDE_GPU" == "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    while IFS=, read -r pid _rest; do
      pid="${pid//[[:space:]]/}"
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      if kill -0 "$pid" 2>/dev/null; then
        add_pid "$pid"
      else
        INVISIBLE_GPU_PIDS+=("$pid")
      fi
    done < <(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)
  fi

  if command -v fuser >/dev/null 2>&1; then
    for dev in /dev/nvidia*; do
      [[ -e "$dev" ]] || continue
      while read -r pid; do
        add_pid "$pid"
      done < <(fuser "$dev" 2>/dev/null | tr ' ' '\n' || true)
    done
  fi
fi

for pid in "${!PIDS[@]}"; do
  add_descendants "$pid"
done

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No matching vLLM residue found. port=$PORT pid_file=$PID_FILE pattern=$MODEL_PATTERN"
  exit 0
fi

echo "Matched processes:"
for pid in "${!PIDS[@]}"; do
  ps -o pid=,ppid=,stat=,cmd= -p "$pid" || true
done | sort -n

if [[ ${#INVISIBLE_GPU_PIDS[@]} -gt 0 ]]; then
  echo "GPU PIDs listed by nvidia-smi but not visible in this PID namespace:"
  printf '  %s\n' "${INVISIBLE_GPU_PIDS[@]}"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only. No process was killed."
  exit 0
fi

echo "Sending SIGTERM..."
for pid in "${!PIDS[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

sleep 5

alive=()
for pid in "${!PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    alive+=("$pid")
  fi
done

if [[ ${#alive[@]} -gt 0 && "$SKIP_KILL9" != "1" ]]; then
  echo "Still alive after SIGTERM, sending SIGKILL..."
  for pid in "${alive[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  sleep 1
fi

if [[ -f "$PID_FILE" ]]; then
  pid_in_file="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$pid_in_file" ]] && ! kill -0 "$pid_in_file" 2>/dev/null; then
    rm -f "$PID_FILE"
  fi
fi

echo "Remaining matches:"
ps -eo pid=,ppid=,stat=,cmd= \
  | grep -E "$MODEL_PATTERN|vllm serve" \
  | grep -v grep \
  | grep -v "kill_vllm_residue.sh" || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU compute processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
fi
