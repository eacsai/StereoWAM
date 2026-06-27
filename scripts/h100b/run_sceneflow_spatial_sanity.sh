#!/usr/bin/env bash
# Scene-flow single-model SPATIAL SANITY.
# #4 depth-token KEEP arm + scene-flow head (GR00T DiT future_tokens), warm-start from baseline B.
# Validates: step-0 == baseline (zero-init head), flow loss computed + decreasing, rot180 flip honored,
#            flow_supervised_samples > 0. ~300 steps on one free GPU.
# Authored on 4090d (canonical), transferred to h100. Lives in scripts/h100b/.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root (scripts/h100b/ -> repo)

export FRAMEWORK=QwenGR00T_DepthTokenFFS
export GPUS=1                                   # h100b GPU 1 is free (GPU 0 = #10 Utonia)
export PORT=29740                               # distinct from default 29734
export DATA_ROOT=playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW
export DATA_MIX=libero_spatial_sfstereo_leftprimary   # single suite (only spatial has GT index)
export BS=16
export MAX_STEPS=300
export SAVE_INTERVAL=999999                     # sanity: don't checkpoint

# #4 KEEP arm depth-token config (run_id contains depthtoken_keep -> guard also pins these)
export STRIP_DEPTH=0
export NUM_DEPTH_TOKENS=64
export POOL_HW=8

# warm-start from baseline B (matches the real #4 keep setup); freeze the VLM trunk
export PRETRAINED_CKPT=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints/qwen3p5_0p8b_4suite_stereo_camrope_rightprimary_ourrender_30k/checkpoints/steps_30000_pytorch_model.pt
export FREEZE_MODULES=qwen_vl_interface

# scene-flow head (zero-init -> step-0 must equal baseline)
export SCENE_FLOW=1
export FLOW_INDEX=/mnt/data/wangqiwei/wangqiwei/sceneflow_gt/spatial_index.json
export FLOW_GT_DIR=/mnt/data/wangqiwei/wangqiwei/sceneflow_gt/libero_spatial_openvla_vanilla_stereo_4dgt_sceneflow
export FLOW_GT_ONLY=1
export FLOW_EXPECTED_FLIP=rot180
export FLOW_STEP0_AUDIT=1
export FLOW_STEP0_WARMUP=1
export FLOW_LAMBDA=0.05
export FLOW_GRID=16

export RUN_ID=qwen0p8_groot_depthtoken_keep_sceneflow_spatial_sanity_warmstartB

exec bash scripts/h100b/run_qwen0p8_groot_ffs.sh
