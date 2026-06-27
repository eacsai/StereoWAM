#!/usr/bin/env bash
# FULL 4-suite scene-flow training: #4 depth-token KEEP + scene-flow head, warm-start B,
# STATIC lambda=130 (codex-calibrated: flow grad ~1% of action grad into the shared trunk).
# 4-suite scene-flow GT read per-suite from each dataset's meta/episode_to_sceneflow_sidecar.json
# (wired 2026-06-16), so FLOW_INDEX is left EMPTY (single path can't serve 4 suites).
# Authored on 4090d, transferred to /mnt/data (shared h100a/h100b). Run on h100a GPU0.
# MAX_STEPS / RUN_ID / GPUS overridable via env (for the GPU smoke).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

export FRAMEWORK=QwenGR00T_DepthTokenFFS
export GPUS=${GPUS:-0}
export PORT=${PORT:-29750}
export DATA_ROOT=playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW
export DATA_MIX=libero_all_sfstereo_leftprimary       # ALL 4 suites
export BS=${BS:-32}
export MAX_STEPS=${MAX_STEPS:-30000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-10000}

# #4 KEEP arm depth-token config
export STRIP_DEPTH=0
export NUM_DEPTH_TOKENS=64
export POOL_HW=8

# warm-start from baseline B; freeze VLM trunk
export PRETRAINED_CKPT=/mnt/data/wangqiwei/wangqiwei/starVLA/
export FREEZE_MODULES=qwen_vl_interface

# scene-flow head, STATIC lambda=130
export SCENE_FLOW=1
export FLOW_LAMBDA=${FLOW_LAMBDA:-130}
export FLOW_GRID=16
export FLOW_MASK=dynamic
export FLOW_GT_ONLY=1
export FLOW_EXPECTED_FLIP=rot180
export FLOW_STEP0_AUDIT=1
export FLOW_STEP0_WARMUP=1
# FLOW_INDEX intentionally EMPTY -> each suite uses its meta/episode_to_sceneflow_sidecar.json
# FLOW_GT_DIR intentionally EMPTY -> index records absolute sidecar paths

export RUN_ID=${RUN_ID:-qwen0p8_groot_depthtoken_keep_sceneflow_all4_lambda130_warmstartB_30k}

exec bash scripts/h100b/run_qwen0p8_groot_ffs.sh
