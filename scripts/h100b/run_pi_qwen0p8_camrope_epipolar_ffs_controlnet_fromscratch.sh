#!/bin/bash
# H100b launcher: PI + Qwen3.5-0.8B + cam_rope + epipolar + FFS ControlNet.
# Init from 4090d epipolar 30k ckpt. eff_batch=96 = BS=12 * 2 GPU * GA=4.
set -e
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFS_REPO_DIR=/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo

cd /mnt/data/wangqiwei/wangqiwei/starVLA
export PYTHONPATH=$(pwd):${FFS_REPO_DIR}:${PYTHONPATH:-}

CONDA_VENV=/opt/conda/envs/starvla/bin
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_STEREO_DATA
data_mix=libero_goal_stereo
base_vlm=./playground/Pretrained_models/Qwen3.5-0.8B
ffs_model_path=${FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-pi_qwen0p8_camrope_epipolar_ffs_controlnet_fromscratch_h100b_$(date +%m%d)}

BS=${BS:-12}
MAX_STEPS=${MAX_STEPS:-30000}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29700}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}

# Symlink stereo dataset (codex round-2 MED#5: fail-closed)
STEREO_TARGET=/mnt/data/wangqiwei/wangqiwei/libero_goal_openvla_vanilla_stereo_lerobot
mkdir -p $libero_data_root
LINK_PATH=$libero_data_root/libero_goal
if [ -e "$LINK_PATH" ] || [ -L "$LINK_PATH" ]; then
  if [ -L "$LINK_PATH" ]; then
    current_target=$(readlink -f "$LINK_PATH")
    expected_target=$(readlink -f "$STEREO_TARGET")
    if [ "$current_target" = "$expected_target" ]; then
      echo "[symlink] $LINK_PATH already points to $STEREO_TARGET — ok"
    else
      echo "[symlink] replacing stale symlink at $LINK_PATH (was $current_target, want $STEREO_TARGET)"
      rm -f "$LINK_PATH"
      ln -s "$STEREO_TARGET" "$LINK_PATH"
    fi
  else
    echo "[symlink] FAIL-CLOSED: $LINK_PATH exists but is a real directory/file, not a symlink. Refusing to silently train on wrong data."
    echo "[symlink] please manually remove or fix: $LINK_PATH"
    exit 1
  fi
else
  ln -s "$STEREO_TARGET" "$LINK_PATH"
  echo "[symlink] created $LINK_PATH -> $STEREO_TARGET"
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
mkdir -p ${output_dir}/checkpoints
cp "$0" "${output_dir}/"

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
if [ "${#existing_ckpts[@]}" -gt 0 ]; then
  [ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true) || { echo "refuse: ckpts present, set RESUME=1"; exit 1; }
fi

echo "[launch] PI + Qwen3.5-0.8B + cam_rope + epipolar + FFS-ControlNet (20-30-48 frozen)"
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA=4 = eff_batch=96 (matches 4090d 6-GPU eff=96)"

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
  --framework.qwenvl.stereo_epipolar_mask_enabled true \
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
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_FFS_ControlNet \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
