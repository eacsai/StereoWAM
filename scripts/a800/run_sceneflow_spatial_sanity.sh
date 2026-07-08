#!/usr/bin/env bash
# Scene-flow single-model SPATIAL SANITY (A800).
# #4 depth-token KEEP arm + scene-flow head (GR00T DiT future_tokens), warm-start from baseline B.
# Validates: step-0 == baseline (zero-init head), flow loss computed + decreasing, rot180 flip honored,
#            flow_supervised_samples > 0. ~300 steps on one free GPU.
# Authored + run on A800 (canonical code source = 4090d). Delegates to scripts/a800/run_qwen0p8_groot_ffs.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root (scripts/a800/ -> repo)

export FRAMEWORK=QwenGR00T_DepthTokenFFS
export GPUS=${GPUS:-0}                           # env-overridable; pick a free A800 GPU
export PORT=${PORT:-29740}
export DATA_ROOT=${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW}
export DATA_MIX=${DATA_MIX:-libero_spatial_sfstereo_leftprimary}   # single suite for the sanity
export BS=${BS:-16}
export MAX_STEPS=${MAX_STEPS:-300}
export SAVE_INTERVAL=${SAVE_INTERVAL:-999999}    # sanity: do not checkpoint

# #4 KEEP arm depth-token config
export STRIP_DEPTH=0
export NUM_DEPTH_TOKENS=64
export POOL_HW=8

# warm-start from baseline B (matches the real #4 keep setup); freeze the VLM trunk.
# FAIL-CLOSED: caller must pass PRETRAINED_CKPT to a valid baseline B checkpoint on A800/4090d.
# The sanity is meaningless from a wrong/empty warm-start, so refuse rather than silently mis-init.
export PRETRAINED_CKPT=${PRETRAINED_CKPT-}
export FREEZE_MODULES=${FREEZE_MODULES:-qwen_vl_interface}
if [ -z "${PRETRAINED_CKPT}" ]; then
  echo "[guard] run_sceneflow_spatial_sanity requires PRETRAINED_CKPT set to a baseline B checkpoint dir." >&2
  echo "        e.g. PRETRAINED_CKPT=/home/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/<baselineB>/ bash \$0" >&2
  exit 3
fi

# scene-flow head (zero-init -> step-0 must equal baseline)
export SCENE_FLOW=1
# FLOW_INDEX / FLOW_GT_DIR default EMPTY -> dataloader uses each suite meta/episode_to_sceneflow_sidecar.json.
# Override via env if you have a standalone spatial GT index on this machine.
export FLOW_INDEX=${FLOW_INDEX:-}
export FLOW_GT_DIR=${FLOW_GT_DIR:-}
export FLOW_GT_ONLY=${FLOW_GT_ONLY:-1}
export FLOW_EXPECTED_FLIP=${FLOW_EXPECTED_FLIP:-rot180}
export FLOW_STEP0_AUDIT=${FLOW_STEP0_AUDIT:-1}
export FLOW_STEP0_WARMUP=${FLOW_STEP0_WARMUP:-1}
export FLOW_LAMBDA=${FLOW_LAMBDA:-0.05}
export FLOW_GRID=${FLOW_GRID:-16}

export RUN_ID=${RUN_ID:-qwen0p8_groot_depthtoken_keep_sceneflow_spatial_sanity_warmstartB}

exec bash scripts/a800/run_qwen0p8_groot_ffs.sh
