#!/bin/bash
# Stereo policy server. Same as mono LIBERO: framework.predict_action(examples).
# Plain QwenPI takes example["image"] = list of PILs directly — VLM's
# build_qwenvl_inputs handles multi-view input via prompt-level concat.
#
# Required env vars (override on command line):
#   CKPT      — path to a SSF checkpoint .pt (X / Y run)
#   GPU_ID    — CUDA device id (default 0)
#   PORT      — websocket port (default 6694)
set -e

cd /data/wangqiwei/ICLR2026/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

CKPT=${CKPT:?set CKPT to a SSF .pt file}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-6694}

# Use the starVLA training venv.
STARVLA_PYTHON=/data/wangqiwei/ICLR2026/starVLA/.venv/bin/python

CUDA_VISIBLE_DEVICES=${GPU_ID} ${STARVLA_PYTHON} \
    deployment/model_server/server_policy.py \
    --ckpt_path ${CKPT} \
    --port ${PORT} \
    --use_bf16
