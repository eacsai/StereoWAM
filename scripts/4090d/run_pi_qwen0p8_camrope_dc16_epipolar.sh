#!/bin/bash
# PI + Qwen0.8B + cam_rope d_c=16 + EPIPOLAR mask.
#
# Background: Phase 2 results gave Mono 82% / Stereo 80% on libero_goal — stereo
# gain was within noise. Phase 3 B injects an explicit geometric prior into the
# 6 standard attention layers of Qwen3.5-0.8B: each token gets a d_c=16-dim
# camera-conditioned rotation (I_{d_c/4} ⊗ P_t), so the attention dot product
# between left and right tokens encodes the relative camera projection P_left^-1 P_right.
# Zero Init (W_q_cam = W_k_cam = 0) preserves baseline behavior at step 0;
# the model gradually activates the camera signal as gradients accumulate.
#
# Hardware: 4090d, 6 GPUs (per Phase 2 Stereo Path B setup).
# Data: libero_goal_stereo_openvla (OpenVLA convention, gripper {0, 1}).
#
# Code branch: phase3b_stereo_camrope (HEAD d4d8895 + codex round-1 fixes).

set -e

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens20f0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export WANDB_MODE=disabled

cd /data/wangqiwei/ICLR2026/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_STEREO_DATA
data_mix=libero_goal_stereo

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-goal_pi_qwen0p8_camrope_dc16_epipolar_$(date +%m%d)}

GPUS=${GPUS:-1,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-6}
PORT=${PORT:-29681}
BS=${BS:-16}
MAX_STEPS=${MAX_STEPS:-30000}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

# Resume guard
RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
if [ "${#existing_ckpts[@]}" -gt 0 ]; then
  if [ "${RESUME:-0}" = "1" ]; then
    echo "[resume] ${#existing_ckpts[@]} valid ckpt(s), RESUME=1 → --trainer.is_resume true"
    RESUME_FLAG=(--trainer.is_resume true)
  else
    echo "[refuse] ${output_dir}/checkpoints non-empty. RESUME=1 to continue or rm -rf to restart." >&2
    exit 1
  fi
fi

# B-1b pre-flight (codex round-1 fix): validate dataloader image order matches
# StereoCamRoPE cam_id 0/1 assumption (primary @ idx 0, right @ 1).
# Refuses to launch if data_config drift breaks the contract — mirrors A2 pattern.
echo "[launch] pre-flight: validate stereo dataloader image order"
.venv/bin/python scripts/4090d/validate_stereo_dataloader.py \
    --data-mix ${data_mix} --num-cameras 2

# B-2 unit tests (codex round-1 fix): run focused tests before long run.
# Validates per_token_cam_id, zero-init parity, signal-flow.
echo "[launch] pre-flight: unit tests on stereo_cam_rope"
.venv/bin/python scripts/4090d/test_phase3b_camrope.py

echo "[launch] Phase 3 B stereo_cam_rope run_id=${run_id}"
echo "[launch] data_mix=${data_mix} BS=${BS} MAX_STEPS=${MAX_STEPS}"

CUDA_VISIBLE_DEVICES=${GPUS} .venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled true \
  --framework.qwenvl.stereo_cam_rope_d_c 16 \
  --framework.qwenvl.stereo_epipolar_mask_enabled true \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Epipolar \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
