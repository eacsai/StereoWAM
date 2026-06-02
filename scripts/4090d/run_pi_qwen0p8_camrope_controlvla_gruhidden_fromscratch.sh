#!/bin/bash
# 4090d launcher: PI + Qwen3.5-0.8B + cam_rope + FFS ControlVLA-style branch,
# injecting the FFS GRU-HIDDEN state net[0] (post-cost-volume, disparity-aware)
# instead of the monocular backbone feature pyramid. Fromscratch, no warm-start.
#
# Motivation (2026-05-29 audit): the backbone-feature variants inject MONOCULAR
# features (FoundationStereo's Feature module is per-view, pre-cost-volume), which
# explains stereo == mono on libero_goal. net[0] carries the actual left-right
# matching / disparity signal. See memory project_ffs_features_are_monocular.
#
# eff_batch = BS(16) * 6 GPU * GA(1) = 96 (matches the 4090d no-epi fromscratch baseline).
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
# codex round-1 MED#3: pin FFS ckpt hash so a stale/wrong file can't silently
# redefine the stereo feature source (and avoid the weights_only=False warning path).
ffs_sha256=98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-pi_qwen0p8_camrope_controlvla_gruhidden_fromscratch_4090d_$(date +%m%d)}

BS=${BS:-16}
MAX_STEPS=${MAX_STEPS:-30000}
GPUS=${GPUS:-2,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-6}
PORT=${PORT:-29704}
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
  [ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true) || { echo "refuse: ckpts present, set RESUME=1"; exit 1; }
fi

echo "[launch] PI + Qwen3.5-0.8B + cam_rope + ControlVLA branch, FFS feature_source=GRU_HIDDEN net[0] (frozen FFS, no epipolar, fromscratch)"
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA=1 = eff_batch=96  GPUS=$GPUS"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name "QwenPIControlVLAFFS" \
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
  --framework.action_model.diffusion_model_cfg.interleave_self_attention false \
  --framework.ffs_controlvla.ffs_model_path ${ffs_model_path} \
  --framework.ffs_controlvla.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_controlvla.ffs_feature_source gru_hidden \
  --framework.ffs_controlvla.ffs_scale 0 \
  --framework.ffs_controlvla.ffs_image_size 256 \
  --framework.ffs_controlvla.ffs_pool_size 8 \
  --framework.ffs_controlvla.use_init_disp false \
  --framework.ffs_controlvla.primary_idx 0 \
  --framework.ffs_controlvla.right_view_idx 1 \
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
  --wandb_project starVLA_FFS_ControlVLA \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
