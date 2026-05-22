#!/usr/bin/env bash
# SSF M1 smoke test — single GPU, 20 steps, wandb disabled.
# Goal: verify QwenPI forward on stereo data does not crash. (SSF cons_loss design dropped 2026-05-18.)
set -euo pipefail

cd "$(dirname "$0")/../../.."   # → repo root
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
export HF_ENDPOINT=https://hf-mirror.com

config_yaml=./examples/LIBERO_STEREO/train_files/mono_replay_libero.yaml
run_id=mono_baseline_$(date +%m%d_%H%M)
run_root_dir=./playground/Checkpoints
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

.venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes 4 \
  --main_process_port 29591 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vlm_data.per_device_batch_size 1 \
  --trainer.max_train_steps 500 \
  --trainer.save_interval 250 \
  --trainer.eval_interval 5000 \
  --trainer.freeze_modules "" \
  --datasets.vla_data.obs_image_size 128 \
  --trainer.logging_frequency 10 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project mono_baseline \
  --wandb_entity disabled \
  \
  2>&1 | tee ${output_dir}/smoke.log
