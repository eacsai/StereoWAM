#!/usr/bin/env bash
# Multi-GPU RoboCasa-365 target50 evaluator for one starVLA checkpoint.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ----------------------------- USER CONFIG -----------------------------
CKPT=${1:-${CKPT:-}}
GPU_LIST=${GPU_LIST:-auto}                  # auto, or e.g. "0 2 5"
MIN_FREE_GB=${MIN_FREE_GB:-20}
BASE_PORT=${BASE_PORT:-18000}
PORT_STRATEGY=${PORT_STRATEGY:-fail}        # fail or next
N_EPISODES=${N_EPISODES:-50}
N_ACT=${N_ACT:-8}
STARVLA_PYTHON=${STARVLA_PYTHON:-/data/wangqiwei/ICLR2026/starVLA/.venv/bin/python}
ROBOCASA365_PYTHON=${ROBOCASA365_PYTHON:-/data/wangqiwei/ICLR2026/robocasa/.venv/bin/python}
TASK_FILTER=${TASK_FILTER:-all}             # all, atomic, composite_seen, composite_unseen
TASK_LIMIT=${TASK_LIMIT:-0}                 # dev-only smoke limit
MODE=${MODE:-leaderboard}                   # leaderboard or dev
REQUIRE_DOWNLOADED=${REQUIRE_DOWNLOADED:-0} # dev-only
SERVER_TIMEOUT=${SERVER_TIMEOUT:-600}
# -----------------------------------------------------------------------

if [[ -z "${CKPT}" ]]; then
  cat >&2 <<USAGE
Usage:
  CKPT=/path/to/checkpoints/steps_...pt bash ${BASH_SOURCE[0]}
  bash ${BASH_SOURCE[0]} /path/to/checkpoints/steps_...pt

Important env vars:
  GPU_LIST=auto|"0 2"  MIN_FREE_GB=20  BASE_PORT=18000  PORT_STRATEGY=fail|next
  N_EPISODES=50        N_ACT=8         TASK_FILTER=all  MODE=leaderboard|dev
  STARVLA_PYTHON=/data/wangqiwei/ICLR2026/starVLA/.venv/bin/python
  ROBOCASA365_PYTHON=/data/wangqiwei/ICLR2026/robocasa/.venv/bin/python
USAGE
  exit 2
fi

EVAL_DIR=${EVAL_DIR:-"$(python3 -c 'import sys,pathlib; print(pathlib.Path(sys.argv[1]).with_suffix(".eval"))' "${CKPT}")"}
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-"${EVAL_DIR}/logs/auto_eval_${RUN_ID}"}
TASK_TSV=${TASK_TSV:-"${EVAL_DIR}/target_tasks_${TASK_FILTER}.tsv"}
ASSIGNMENT_TSV=${ASSIGNMENT_TSV:-"${EVAL_DIR}/assignment_${RUN_ID}.tsv"}
mkdir -p "${EVAL_DIR}" "${LOG_DIR}"

SERVER_PIDS=()
ACTIVE_GPUS=()
ACTIVE_PORTS=()

cleanup_servers() {
  local pid
  if ((${#SERVER_PIDS[@]} > 0)); then
    echo "[cleanup] stopping policy servers: ${SERVER_PIDS[*]}"
  fi
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup_servers EXIT
trap 'cleanup_servers; trap - EXIT INT TERM; exit 130' INT TERM

run_python() {
  "${ROBOCASA365_PYTHON}" "$@"
}

echo "=== RoboCasa-365 Auto Eval ==="
echo "ckpt                 : ${CKPT}"
echo "eval_dir             : ${EVAL_DIR}"
echo "log_dir              : ${LOG_DIR}"
echo "mode/task_filter     : ${MODE}/${TASK_FILTER}"
echo "episodes/action_steps: ${N_EPISODES}/${N_ACT}"
echo "gpu_list/min_free_gb : ${GPU_LIST}/${MIN_FREE_GB}"
echo "base_port            : ${BASE_PORT} (${PORT_STRATEGY})"
echo "starVLA python       : ${STARVLA_PYTHON}"
echo "robocasa python      : ${ROBOCASA365_PYTHON}"

generate_task_table() {
  local args=(
    "${SCRIPT_DIR}/gen_target_tasks.py"
    --output "${TASK_TSV}"
    --task-filter "${TASK_FILTER}"
    --mode "${MODE}"
    --no-header
  )
  if [[ "${TASK_LIMIT}" != "0" ]]; then
    args+=(--limit "${TASK_LIMIT}")
  fi
  if [[ "${REQUIRE_DOWNLOADED}" == "1" ]]; then
    args+=(--require-downloaded)
  fi
  run_python "${args[@]}"
}

pending_count() {
  run_python - "${TASK_TSV}" "${CKPT}" "${N_EPISODES}" <<'PY'
import csv
import json
import sys
from pathlib import Path

task_tsv, ckpt, n_episodes_s = sys.argv[1:4]
n_episodes = int(n_episodes_s)
eval_dir = Path(ckpt).with_suffix(".eval")

def complete(env_name: str) -> bool:
    path = eval_dir / f"{env_name.replace('/', '_')}.json"
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return False
    successes = data.get("successes")
    return isinstance(successes, list) and len(successes) == n_episodes

pending = 0
with open(task_tsv, newline="") as f:
    for split, task_name, horizon in csv.reader(f, delimiter="\t"):
        env_name = task_name if task_name.startswith("robocasa/") else f"robocasa/{task_name}"
        if not complete(env_name):
            pending += 1
print(pending)
PY
}

pick_gpus() {
  if [[ "${GPU_LIST}" != "auto" ]]; then
    printf '%s\n' ${GPU_LIST}
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu] nvidia-smi not found and GPU_LIST=auto" >&2
    return 1
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    awk -F',' -v min_gb="${MIN_FREE_GB}" '
      BEGIN { min_mb = min_gb * 1024 }
      {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        if (($2 + 0) >= min_mb) print $1
      }'
}

port_in_use() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN -Pn >/dev/null 2>&1
    return $?
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk -v port="${port}" '
      NR > 1 {
        n = split($4, parts, ":")
        if (parts[n] == port) found = 1
      }
      END { exit found ? 0 : 1 }'
    return $?
  fi
  "${STARVLA_PYTHON}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    sys.exit(0)
finally:
    sock.close()
sys.exit(1)
PY
}

port_is_free() {
  ! port_in_use "$1"
}

choose_port() {
  local gpu_id=$1
  local candidate=$((BASE_PORT + gpu_id))
  if port_is_free "${candidate}"; then
    echo "${candidate}"
    return 0
  fi
  if [[ "${PORT_STRATEGY}" != "next" ]]; then
    echo "[port] ${candidate} is already in use for gpu ${gpu_id}; set PORT_STRATEGY=next to scan" >&2
    return 1
  fi
  for ((candidate = BASE_PORT + 100; candidate < BASE_PORT + 2000; candidate++)); do
    if port_is_free "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "[port] no free port found near BASE_PORT=${BASE_PORT}" >&2
  return 1
}

port_accepting() {
  local port=$1
  "${STARVLA_PYTHON}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

server_handshake_ok() {
  local port=$1
  local expected_ckpt=$2
  "${STARVLA_PYTHON}" - "${port}" "${expected_ckpt}" <<'PY'
import json
import os
import sys

import websockets.sync.client

from deployment.model_server.tools import msgpack_numpy

port = int(sys.argv[1])
expected_ckpt = sys.argv[2]
for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(key, None)

conn = websockets.sync.client.connect(
    f"ws://127.0.0.1:{port}",
    compression=None,
    max_size=None,
    open_timeout=5,
    ping_interval=None,
    ping_timeout=5,
)
try:
    metadata = msgpack_numpy.unpackb(conn.recv())
finally:
    conn.close()

got = metadata.get("ckpt_path")
if got != expected_ckpt:
    print(
        f"ckpt_path mismatch: expected {expected_ckpt!r}, got {got!r}; metadata={metadata!r}",
        file=sys.stderr,
    )
    sys.exit(3)
print(json.dumps(metadata, sort_keys=True))
PY
}

wait_for_server() {
  local gpu_id=$1
  local port=$2
  local pid=$3
  local server_log=$4
  local deadline=$((SECONDS + SERVER_TIMEOUT))
  local meta

  echo "[server] waiting for gpu=${gpu_id} port=${port} pid=${pid}"
  while ((SECONDS < deadline)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[server] pid ${pid} exited before readiness. Tail of ${server_log}:" >&2
      tail -n 80 "${server_log}" >&2 || true
      return 1
    fi
    if port_accepting "${port}"; then
      if meta=$(server_handshake_ok "${port}" "${CKPT}" 2>&1); then
        echo "[server] ready gpu=${gpu_id} port=${port} pid=${pid} metadata=${meta}"
        return 0
      fi
      echo "[server] handshake failed gpu=${gpu_id} port=${port}: ${meta}" >&2
      if [[ "${meta}" == *"ckpt_path mismatch"* ]]; then
        return 1
      fi
    fi
    sleep 5
  done
  echo "[server] timeout waiting for gpu=${gpu_id} port=${port} pid=${pid}. Tail of ${server_log}:" >&2
  tail -n 80 "${server_log}" >&2 || true
  return 1
}

start_servers() {
  local selected_gpus=()
  local gpu_id port server_log pid
  mapfile -t selected_gpus < <(pick_gpus)
  if ((${#selected_gpus[@]} == 0)); then
    echo "[gpu] no GPUs selected" >&2
    return 1
  fi
  echo "[gpu] selected: ${selected_gpus[*]}"

  for gpu_id in "${selected_gpus[@]}"; do
    port=$(choose_port "${gpu_id}") || {
      echo "[server] skipping gpu=${gpu_id} because no safe port is available" >&2
      continue
    }
    server_log="${LOG_DIR}/server_gpu${gpu_id}_port${port}.log"
    echo "[server] starting gpu=${gpu_id} port=${port} log=${server_log}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT}" \
        --port "${port}" \
        --use_bf16 \
        --idle_timeout -1 \
        >"${server_log}" 2>&1 &
    pid=$!
    SERVER_PIDS+=("${pid}")
    if wait_for_server "${gpu_id}" "${port}" "${pid}" "${server_log}"; then
      ACTIVE_GPUS+=("${gpu_id}")
      ACTIVE_PORTS+=("${port}")
    else
      echo "[server] discarding gpu=${gpu_id} port=${port} pid=${pid}" >&2
      kill "${pid}" 2>/dev/null || true
    fi
  done

  if ((${#ACTIVE_GPUS[@]} == 0)); then
    echo "[server] no usable policy servers" >&2
    return 1
  fi
  echo "[server] active gpu/port pairs:"
  for i in "${!ACTIVE_GPUS[@]}"; do
    echo "  gpu ${ACTIVE_GPUS[$i]} -> port ${ACTIVE_PORTS[$i]}"
  done
}

make_assignment() {
  run_python - "${TASK_TSV}" "${CKPT}" "${N_EPISODES}" "${ASSIGNMENT_TSV}" \
    "${ACTIVE_GPUS[*]}" "${ACTIVE_PORTS[*]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

task_tsv, ckpt, n_episodes_s, out_tsv, gpus_s, ports_s = sys.argv[1:7]
n_episodes = int(n_episodes_s)
gpus = gpus_s.split()
ports = ports_s.split()
if len(gpus) != len(ports):
    raise SystemExit(f"gpu/port count mismatch: {gpus} vs {ports}")
if not gpus:
    raise SystemExit("no active GPUs")

eval_dir = Path(ckpt).with_suffix(".eval")

def complete(env_name: str) -> bool:
    path = eval_dir / f"{env_name.replace('/', '_')}.json"
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return False
    successes = data.get("successes")
    return isinstance(successes, list) and len(successes) == n_episodes

tasks = []
skipped = []
with open(task_tsv, newline="") as f:
    for split, task_name, horizon_s in csv.reader(f, delimiter="\t"):
        horizon = int(horizon_s)
        env_name = task_name if task_name.startswith("robocasa/") else f"robocasa/{task_name}"
        if complete(env_name):
            skipped.append(env_name)
            continue
        tasks.append((split, task_name, env_name, horizon))

bins = [{"gpu": gpu, "port": port, "cost": 0, "tasks": []} for gpu, port in zip(gpus, ports)]
for task in sorted(tasks, key=lambda item: item[3], reverse=True):
    target = min(bins, key=lambda item: (item["cost"], int(item["gpu"])))
    target["tasks"].append(task)
    target["cost"] += task[3] * n_episodes

with open(out_tsv, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    for item in bins:
        for split, task_name, env_name, horizon in item["tasks"]:
            writer.writerow([item["gpu"], item["port"], split, task_name, env_name, horizon])

print(f"[assign] skipped complete tasks: {len(skipped)}")
print(f"[assign] pending tasks: {len(tasks)}")
for item in bins:
    print(
        f"[assign] gpu {item['gpu']} port {item['port']}: "
        f"{len(item['tasks'])} tasks, cost {item['cost']}"
    )
PY
}

run_worker() {
  local worker_gpu=$1
  local worker_port=$2
  local assigned_gpu assigned_port split task_name env_name horizon
  echo "[worker ${worker_gpu}] starting on port ${worker_port}"
  while IFS=$'\t' read -r assigned_gpu assigned_port split task_name env_name horizon; do
    [[ -n "${assigned_gpu:-}" ]] || continue
    [[ "${assigned_gpu}" == "${worker_gpu}" ]] || continue
    echo "[worker ${worker_gpu}] task=${env_name} split=${split} horizon=${horizon}"
    N_EPISODES="${N_EPISODES}" \
    N_ACT="${N_ACT}" \
    ROBOCASA365_PYTHON="${ROBOCASA365_PYTHON}" \
    LOG_DIR="${LOG_DIR}" \
      bash "${SCRIPT_DIR}/eval_robocasa365_one.sh" "${CKPT}" "${env_name}" "${horizon}" "${worker_gpu}" "${worker_port}"
  done <"${ASSIGNMENT_TSV}"
  echo "[worker ${worker_gpu}] done"
}

run_workers() {
  local worker_pids=()
  local pid status=0
  local i
  for i in "${!ACTIVE_GPUS[@]}"; do
    run_worker "${ACTIVE_GPUS[$i]}" "${ACTIVE_PORTS[$i]}" &
    worker_pids+=("$!")
  done
  echo "[workers] pids: ${worker_pids[*]}"
  for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
      echo "[workers] worker pid ${pid} failed" >&2
      status=1
    fi
  done
  return "${status}"
}

run_aggregator() {
  run_python "${SCRIPT_DIR}/aggregate_robocasa365.py" \
    --ckpt "${CKPT}" \
    --tasks-tsv "${TASK_TSV}" \
    --n-episodes "${N_EPISODES}"
}

generate_task_table
PENDING=$(pending_count)
echo "[resume] pending tasks: ${PENDING}"

if [[ "${PENDING}" == "0" ]]; then
  echo "[resume] all tasks already have ${N_EPISODES} successes; aggregating only"
  run_aggregator
  trap - EXIT INT TERM
  exit 0
fi

start_servers
make_assignment

status=0
if ! run_workers; then
  status=1
fi

cleanup_servers
trap - EXIT INT TERM

if ! run_aggregator; then
  status=1
fi

exit "${status}"
