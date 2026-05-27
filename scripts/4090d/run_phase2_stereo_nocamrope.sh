#!/bin/bash
# Phase 2 Stereo-SelfRender (NO cam_rope) — fair ablation vs Phase 3 B Stereo+CamRoPE.
# Identical config to phase3b (same data, same eff_batch=96, same 30k steps) EXCEPT:
#   - stereo_cam_rope_enabled false (no camera-frame RoPE injection)
# Goal: paper table "cam_rope ablation" — 30k step Stereo SR with vs without cam_rope.
# Hardware: GPU 1,3,4,5,6,7 (avoid GPU 0,2 occupied; GPU 1 partially used by other user but OK).
# eff_batch = BS 16 * 6 GPU = 96, identical to phase3b.

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

Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_STEREO_DATA
data_mix=libero_goal_stereo

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-goal_phase2_stereo_nocamrope_$(date +%m%d)}

GPUS=${GPUS:-1,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-6}
PORT=${PORT:-29682}
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
    echo "[refuse] ${output_dir}/checkpoints non-empty. RESUME=1 to continue or rm -rf to restart." >&2
    exit 1
  fi
fi

# Pre-flight: validate stereo dataloader image order (cam_id 0/1 ordering).
# Same data shape as phase3b — must pass even though cam_rope is disabled.
echo "[launch] pre-flight: validate stereo dataloader image order"
.venv/bin/python scripts/4090d/validate_stereo_dataloader.py \
    --data-mix ${data_mix} --num-cameras 2

echo "[launch] Phase 2 Stereo NO cam_rope (ablation) run_id=${run_id}"
echo "[launch] data_mix=${data_mix} BS=${BS} NUM_PROCESSES=${NUM_PROCESSES} MAX_STEPS=${MAX_STEPS}"

CUDA_VISIBLE_DEVICES=${GPUS} .venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled false \
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
  --wandb_project starVLA_Phase2Ablation \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
