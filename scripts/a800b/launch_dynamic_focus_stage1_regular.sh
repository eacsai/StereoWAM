#!/usr/bin/env bash
# Queue entry for a800b: stage1-only dynamic-focused scene-flow, regular past-flow gate.
set -euo pipefail
export REPO_DIR=${REPO_DIR:-/home/wangqiwei/ICLR2026/starVLA}
export RUN_ID=${RUN_ID:-qwen0p8_groot_cascade_motion_cambranch_pastflow_dynamic_focus_stage1_fromscratch_leftprimary}
export GPUS=${GPUS:-0}
export PORT=${PORT:-29861}
export PAST_FLOW_FULL_INJECT=0
export STAGE=learn_scene_flow
exec "${REPO_DIR}/scripts/a800/run_sceneflow_dynamic_focus_stage1.sh" "$@"
