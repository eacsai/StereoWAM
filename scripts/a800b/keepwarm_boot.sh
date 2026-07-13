#!/usr/bin/env bash
set -euo pipefail
export REPO=${REPO:-/home/wangqiwei/ICLR2026/starVLA}
export QUEUE_FILE=${QUEUE_FILE:-${REPO}/scripts/a800b/keepwarm_queue.txt}
export STATE_DIR=${STATE_DIR:-${REPO}/playground/Checkpoints/keepwarm_state_a800b}
export POLL=${POLL:-180}
export MEM_FREE_MIB=${MEM_FREE_MIB:-1500}
export LAUNCH_GRACE=${LAUNCH_GRACE:-600}
export PRUNE_MODE=${PRUNE_MODE:-off}
export FINAL_CKPT_GLOB=${FINAL_CKPT_GLOB:-steps_30000_pytorch_model.pt}
mkdir -p "${STATE_DIR}"
exec bash "${REPO}/scripts/a800b/keep_warm_queue_runner.sh"
