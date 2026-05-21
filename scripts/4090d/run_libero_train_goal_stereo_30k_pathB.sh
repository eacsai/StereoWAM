#!/bin/bash
# Stereo Path B baseline: patched self-rendered stereo data + Stereo-SelfRender
# (primary + right_view 6cm baseline). Trained on 4090d 6 GPUs to match Mono-Official
# setup exactly (eff batch 96, 30k steps, 2.88M samples) — strictest within-cluster
# Mono-Official vs Stereo-Path-B comparison for paper.
#
# Prerequisite:
#   1. patch_self_rendered_parquets.py applied in-place (gripper {-1,+1} → {0,1})
#   2. playground/Datasets/LEROBOT_LIBERO_STEREO_DATA → libero_goal_stereo_openvla
#
# After training:
#   - eval with DEFAULT openvla gripper convention (no --args.gripper-convention flag)
#   - direct compare with Mono-Official 30k peak 94%

set -e

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens20f0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export WANDB_MODE=disabled

cd /data/wangqiwei/ICLR2026/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

Framework_name=QwenPI
freeze_module_list=""
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_STEREO_DATA   # → libero_goal_stereo_openvla (patched)
data_mix=libero_goal_stereo                                        # primary + right_view
run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-goal_stereo_30k_pathB_0521}

GPUS=${GPUS:-0,1,2,3,4,5}
NUM_PROCESSES=${NUM_PROCESSES:-6}
PORT=${PORT:-29679}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# resume guard
RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
if [ "${#existing_ckpts[@]}" -gt 0 ]; then
  if [ "${RESUME:-0}" = "1" ]; then
    echo "[resume] --trainer.is_resume true"
    RESUME_FLAG=(--trainer.is_resume true)
  else
    echo "[refuse] ${output_dir}/checkpoints non-empty. RESUME=1 / RUN_ID=other / rm -rf checkpoints" >&2
    exit 1
  fi
elif [ "${RESUME:-0}" = "1" ]; then
  echo "[refuse] RESUME=1 but no valid steps_*_pytorch_model.pt. Aborting." >&2
  exit 1
fi

echo "[launch] data_mix=${data_mix}  run_id=${run_id}  data_root=${libero_data_root}"

CUDA_VISIBLE_DEVICES=${GPUS} .venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_StereoVLA \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
