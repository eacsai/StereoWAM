#!/bin/bash
# Phase 3 E: Stereo Camera-Frame RoPE d_c=64 ablation (StereoWorld paper sweet spot D/4).
# Base = Qwen3.5-0.8B (head_dim=256, D/4=64). Same as phase3b but d_c=64 instead of 16.
# Goal: paper "d_c sweep" ablation — 30k step Stereo SR for d_c ∈ {16 (phase3b), 32 (phase3d), 64 (phase3e)}.
# Hardware: 2× H100 80GB on jinshan_dev container. BS=48 × 2 GPU = 96 eff_batch (fair vs phase3b 16×6=96).

set -e

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled

cd /mnt/data/wangqiwei/wangqiwei/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

CONDA_VENV=/opt/conda/envs/starvla/bin
Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_STEREO_DATA
data_mix=libero_goal_stereo

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-goal_phase3e_camrope_dc64_$(date +%m%d)}

GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29683}
BS=${BS:-48}
MAX_STEPS=${MAX_STEPS:-30000}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

# Resume guard (mirrors phase3b pattern)
RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
if [ "${#existing_ckpts[@]}" -gt 0 ]; then
  if [ "${RESUME:-0}" = "1" ]; then
    echo "[resume] ${#existing_ckpts[@]} valid ckpt(s), RESUME=1 -> --trainer.is_resume true"
    RESUME_FLAG=(--trainer.is_resume true)
  else
    echo "[refuse] ${output_dir}/checkpoints non-empty. RESUME=1 to continue or rm -rf to restart." >&2
    exit 1
  fi
fi

# Pre-flight: same as phase3b (validate stereo dataloader + unit tests on cam_rope code)
echo "[launch] pre-flight: validate stereo dataloader image order"
${CONDA_VENV}/python scripts/4090d/validate_stereo_dataloader.py \
    --data-mix ${data_mix} --num-cameras 2

echo "[launch] pre-flight: unit tests on stereo_cam_rope"
${CONDA_VENV}/python scripts/4090d/test_phase3b_camrope.py

echo "[launch] Phase 3 E Stereo+CamRoPE d_c=64 (D/4 sweet spot)  run_id=${run_id}"
echo "[launch] base=Qwen3.5-0.8B data_mix=${data_mix} BS=${BS}*${NUM_PROCESSES}GPU=eff_batch_$((BS*NUM_PROCESSES)) MAX_STEPS=${MAX_STEPS}"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled true \
  --framework.qwenvl.stereo_cam_rope_d_c 64 \
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
  --wandb_project starVLA_DCSweep \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
