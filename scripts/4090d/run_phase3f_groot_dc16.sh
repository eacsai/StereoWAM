#!/bin/bash
# Phase 3 F: GR00T action head (DiT-B) + Qwen3.5-0.8B + Stereo + CamRoPE d_c=16.
# Paper-fair ablation vs Phase 3 B (PI + 0.8B + d_c=16 = 94%): only action head differs.
# Same BS=16 * 6 GPU = 96 eff_batch, same 30k step, same stereo data, same Zero Init.

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

# Note: framework.name defaults to QwenGR00T in starvla_cotrain_libero.yaml.
# Phase 3 B uses --framework.name QwenPI override; we drop that override here
# so the yaml default QwenGR00T is used.
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_STEREO_DATA
data_mix=libero_goal_stereo

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-goal_phase3f_groot_dc16_$(date +%m%d)}

GPUS=${GPUS:-1,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-6}
PORT=${PORT:-29686}
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
    echo "[resume] ${#existing_ckpts[@]} valid ckpt(s), RESUME=1 -> --trainer.is_resume true"
    RESUME_FLAG=(--trainer.is_resume true)
  else
    echo "[refuse] ${output_dir}/checkpoints non-empty. RESUME=1 or rm -rf to restart." >&2
    exit 1
  fi
fi

# Pre-flight: validate stereo dataloader (same as phase3b)
echo "[launch] pre-flight: validate stereo dataloader image order"
.venv/bin/python scripts/4090d/validate_stereo_dataloader.py \
    --data-mix ${data_mix} --num-cameras 2

echo "[launch] Phase 3 F GR00T+CamRoPE d_c=16 (Qwen3.5-0.8B)  run_id=${run_id}"
echo "[launch] data_mix=${data_mix} BS=${BS} NUM_PROCESSES=${NUM_PROCESSES} eff_batch=$((BS*NUM_PROCESSES)) MAX_STEPS=${MAX_STEPS}"

CUDA_VISIBLE_DEVICES=${GPUS} .venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled true \
  --framework.qwenvl.stereo_cam_rope_d_c 16 \
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
  --wandb_project starVLA_GR00T_ablation \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
