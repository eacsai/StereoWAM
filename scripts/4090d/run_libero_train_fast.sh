#!/bin/bash
# Fast smoke-validation variant of run_libero_train_4090d.sh:
#   - libero_goal single suite (smallest, 428 episodes)
#   - 5000 train steps (paper does 30k joint, we just need to see codebase converges)
#   - save_interval 1000 (5 ckpts for convergence curve)
#   - eval_interval 999999 (skip online eval entirely)
# Everything else identical to run_libero_train_4090d.sh.
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
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_goal
run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-fast_smoke_goal_qwenpi_0519}

GPUS=${GPUS:-0,2,3,4,5,6}
NUM_PROCESSES=${NUM_PROCESSES:-6}
PORT=${PORT:-29675}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# --- Codex finding 2: resume / overwrite guard (see run_libero_train.sh) ---
RESUME_FLAG=()
if [ -d "${output_dir}/checkpoints" ] && [ -n "$(ls -A "${output_dir}/checkpoints" 2>/dev/null)" ]; then
  if [ "${RESUME:-0}" = "1" ]; then
    echo "[resume] existing ckpts in ${output_dir}/checkpoints, RESUME=1 → passing --trainer.is_resume true"
    RESUME_FLAG=(--trainer.is_resume true)
  else
    echo "[refuse] ${output_dir}/checkpoints is non-empty. Pick one:" >&2
    echo "  RESUME=1 bash $0   |   RUN_ID=${run_id}_v2 bash $0   |   rm -rf ${output_dir}/checkpoints" >&2
    exit 1
  fi
fi
# --------------------------------------------------------------------------

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
  --trainer.max_train_steps 5000 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
