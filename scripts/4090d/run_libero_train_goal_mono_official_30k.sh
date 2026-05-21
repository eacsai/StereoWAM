#!/bin/bash
# Z1 baseline: official IPEC LeRobot data + mono primary only (NO WRIST) + libero_goal × 30000 steps.
# Same step budget and hyperparams as goal_only_30k_qwenpi_0519 (the wrist+primary 30k=94% run),
# only difference: data_mix swapped to libero_goal_mono_official (drops wrist).
# Lets us isolate "wrist contributes how much" — paper Z reference baseline.
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
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA  # official IPEC
data_mix=libero_goal_mono_official                         # NEW: drop wrist, official data
run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-goal_only_30k_mono_primary_official_0520}

GPUS=${GPUS:-0,1,2,3,4,5}   # avoid GPU 7 (busy with someone else)
NUM_PROCESSES=${NUM_PROCESSES:-6}
PORT=${PORT:-29678}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# resume guard (codex F3 fix verbatim)
RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
if [ "${#existing_ckpts[@]}" -gt 0 ]; then
  if [ "${RESUME:-0}" = "1" ]; then
    echo "[resume] passing --trainer.is_resume true"
    RESUME_FLAG=(--trainer.is_resume true)
  else
    echo "[refuse] ${output_dir}/checkpoints non-empty. RESUME=1 / RUN_ID=other / rm -rf checkpoints" >&2
    exit 1
  fi
elif [ "${RESUME:-0}" = "1" ]; then
  echo "[refuse] RESUME=1 but no valid steps_*_pytorch_model.pt. Aborting." >&2
  exit 1
fi

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
  --wandb_project starVLA_Libero \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
