#!/bin/bash
# 4090d launcher: PI + Qwen3.5-0.8B + cam_rope (NO epipolar) + FFS ControlNet.
# Init from 4090d cam_rope-only 30k ckpt (Phase 3B baseline = 94%).
# eff_batch=96 = BS=16 * 6 GPU * GA=1 (matches original 4090d epipolar baseline training).
set -e
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFS_REPO_DIR=/data/wangqiwei/ICLR2026/Fast-FoundationStereo

cd /data/wangqiwei/ICLR2026/starVLA
export PYTHONPATH=$(pwd):${FFS_REPO_DIR}:${PYTHONPATH:-}

CONDA_VENV=.venv/bin
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_STEREO_DATA
data_mix=libero_goal_stereo
base_vlm=./playground/Pretrained_models/Qwen3.5-0.8B
ffs_model_path=${FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth
init_ckpt=./playground/Checkpoints/goal_phase3b_camrope_0523/checkpoints/steps_30000_pytorch_model.pt

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-pi_qwen0p8_camrope_NOEpipolar_ffs_controlnet_init_camrope30k_4090d_$(date +%m%d)}

BS=${BS:-16}
MAX_STEPS=${MAX_STEPS:-30000}
GPUS=${GPUS:-1,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-6}
PORT=${PORT:-29701}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
mkdir -p ${output_dir}/checkpoints
cp "$0" "${output_dir}/"

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
if [ "${#existing_ckpts[@]}" -gt 0 ]; then
  [ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true) || { echo refuse; exit 1; }
fi

echo "[launch] PI + Qwen3.5-0.8B + cam_rope (NO epipolar) + FFS-CN (20-30-48 frozen)"
echo "[launch] init from cam_rope-only 30k: ${init_ckpt}"
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA=1 = eff_batch=96"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name "QwenPIControlNetFFS" \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled true \
  --framework.qwenvl.stereo_cam_rope_d_c 16 \
  --framework.qwenvl.stereo_cam_rope_num_cameras 2 \
  --framework.qwenvl.stereo_cam_rope_baseline_m 0.06 \
  --framework.qwenvl.stereo_cam_rope_fovy_degrees 45.0 \
  --framework.qwenvl.stereo_cam_rope_image_width 256 \
  --framework.qwenvl.stereo_cam_rope_image_height 256 \
  --framework.qwenvl.stereo_cam_rope_spatial_merge 2 \
  --framework.qwenvl.stereo_cam_rope_init_mode zero \
  --framework.qwenvl.stereo_epipolar_mask_enabled false \
  --framework.ffs_controlnet.ffs_model_path ${ffs_model_path} \
  --framework.ffs_controlnet.ffs_scale 0 \
  --framework.ffs_controlnet.ffs_image_size 256 \
  --framework.ffs_controlnet.primary_idx 0 \
  --framework.ffs_controlnet.right_view_idx 1 \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 999999 \
  --trainer.pretrained_checkpoint ${init_ckpt} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_FFS_ControlNet_NoEpipolar \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
