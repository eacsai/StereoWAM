#!/bin/bash
# Fresh upstream starVLA reproduction: Qwen2.5-VL-3B + GR00T + libero_goal (mono primary+wrist) on h100b.
# Switch from Qwen3-VL-4B (NaN at step 5k, Issue 171) to Qwen2.5-VL-3B (older, more stable per upstream issues).
# Official recipe: BS=16 GPU=2 GA=4 -> eff_batch=128.

set -e
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/data/wangqiwei/wangqiwei/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

CONDA_VENV=/opt/conda/envs/starvla/bin
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_goal
base_vlm=./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-upstream_groot_qwen2p5vl3b_libero_goal_h100b_$(date +%m%d)}

BS=${BS:-16}
MAX_STEPS=${MAX_STEPS:-30000}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29699}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
[ "${#existing_ckpts[@]}" -gt 0 ] && [ "${RESUME:-0}" != "1" ] && { echo refuse; exit 1; }
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true)

echo "[launch] FRESH UPSTREAM (github HEAD 61ea142): QwenGR00T + Qwen2.5-VL-3B-Instruct"
echo "[launch] data_mix=${data_mix} (libero_franka primary+wrist mono)"
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA=4 = eff_batch_128"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.qwenvl.base_vlm ${base_vlm} \
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
  --wandb_project starVLA_Upstream_Reproduce \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
