#!/bin/bash
# Run a stereo LIBERO closed-loop eval against a running stereo policy server.
#
# Required env vars (override on command line):
#   CKPT             — path to the .pt being evaluated (only used to derive
#                      the output folder name; the actual model is loaded by
#                      the policy server)
#   PORT             — websocket port the policy server is listening on (6694)
#   TASK_SUITE       — libero_spatial / libero_object / libero_goal / libero_10
#   NUM_TRIALS       — trials per task (default 50; use 5 for smoke)
#   MAX_TASKS        — -1 = all (default); set 1 for a single-task smoke
#   STEREO_BASELINE  — must match training (default 0.06 m)
set -e

cd /data/wangqiwei/ICLR2026/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

CKPT=${CKPT:?set CKPT to a SSF .pt}
PORT=${PORT:-6694}
TASK_SUITE=${TASK_SUITE:-libero_spatial}
NUM_TRIALS=${NUM_TRIALS:-50}
MAX_TASKS=${MAX_TASKS:--1}
STEREO_BASELINE=${STEREO_BASELINE:-0.06}

LIBERO_VENV=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python
LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO

export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:${PYTHONPATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
model_root=$(echo "$CKPT" | awk -F'/checkpoints/' '{print $1}')
video_out_path="${model_root}/results/${TASK_SUITE}/${folder_name}"

${LIBERO_VENV} ./examples/LIBERO_STEREO/eval_files/eval_libero_stereo.py \
    --args.host 127.0.0.1 \
    --args.port ${PORT} \
    --args.task-suite-name ${TASK_SUITE} \
    --args.num-trials-per-task ${NUM_TRIALS} \
    --args.max-tasks ${MAX_TASKS} \
    --args.stereo-baseline ${STEREO_BASELINE} \
    --args.video-out-path ${video_out_path} \
    --args.pretrained-path ${CKPT}
