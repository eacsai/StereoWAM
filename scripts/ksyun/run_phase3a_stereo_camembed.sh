#!/bin/bash
# Phase 3 A2: Stereo training with cam_id additive embedding (zero-init).
#
# Tests whether giving the model an explicit (left / right) signal — via a
# learnable embedding added to image token features BEFORE language attention —
# improves stereo VLA performance over the Phase 2 baseline.
#
# Baseline (Phase 2 Path B Stereo, 30k step on libero_goal): 80.0%
# Mono Path B (20k step):                                    82.0%
# This run adds StereoCamEmbedding (cam_id_embed + pos_x/y_embed, all zero-init
# → step 0 byte-identical to mono baseline). Expectation: stereo signal becomes
# explicit, hopefully closing the gap to or beating mono.
#
# Hardware: 2× H100 80GB on jinshan_dev (ksyun container).
#
# Code: branch phase3a_stereo_camembed (off starVLA_dev).
# StereoCamEmbedding module: starVLA/model/modules/stereo/cam_embed.py
# Hook integration: starVLA/model/framework/VLM4A/QwenPI.py

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
run_id=${RUN_ID:-goal_phase3a_camembed_$(date +%m%d)}

BS=${BS:-48}
MAX_STEPS=${MAX_STEPS:-20000}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29680}

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

# A2 fix-1 (codex 2026-05-22): pre-flight validate dataloader image order
#   matches StereoCamEmbedding cam_id 0/1 assumption (primary @ idx 0, right @ 1).
#   Refuses to launch if data_config drift breaks the contract.
echo "[launch] pre-flight: validate stereo dataloader image order"
${CONDA_VENV}/python scripts/4090d/validate_stereo_dataloader.py \
    --data-mix ${data_mix} --num-cameras 2

echo "[launch] Phase3 A2 stereo_cam_embed run_id=${run_id}"
echo "[launch] data_mix=${data_mix} BS=${BS} MAX_STEPS=${MAX_STEPS}"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_embed_enabled true \
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
  --wandb_project starVLA_StereoVLA \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
