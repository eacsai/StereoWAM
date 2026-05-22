#!/bin/bash
# Launch starVLA-paper Qwen3-vl-PI reproduction (mono / 4-suite joint).
# Uses 6 GPUs DDP (GPU 2-7 by default; override via GPUS).
set -e

cd /data/wangqiwei/ICLR2026/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=${GPUS:-0,2,3,4,5,6}
PORT=${PORT:-29672}

CUDA_VISIBLE_DEVICES=${GPUS} .venv/bin/accelerate launch \
  --config_file starVLA/config/accelerate/multi_gpu_bf16.yaml \
  --num_processes 6 \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO_STEREO/train_files/reproduce_qwenpi_mono_4suite.yaml \
  --run_root_dir ./playground/Checkpoints \
  --run_id reproduce_qwenpi_primary_only_0518 \
  --wandb_project reproduce_qwen3pi \
  --wandb_entity disabled
