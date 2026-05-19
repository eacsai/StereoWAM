#!/bin/bash
# Adapted from examples/LIBERO/train_files/run_libero_train.sh for 4090d:
#   - CUDA_VISIBLE_DEVICES=0,2,3,4,5,6 (GPUs 1, 7 owned by other users)
#   - NUM_PROCESSES=6 (matches above)
#   - wandb_entity=disabled (wandb not configured on 4090d)
#   - run_id with date stamp
# Everything else is byte-identical to upstream.
set -e

# Single-node 6×GPU training: NVLink + shared-memory are enough; the
# upstream NCCL_SOCKET_IFNAME=bond0 / NCCL_IB_HCA=mlx5_2,mlx5_3 are
# specific to jye624's multi-node cluster and break on 4090d (only ens20f0
# present, no InfiniBand). Force socket transport + sane defaults instead.
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens20f0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

# Actually disable wandb (the --wandb_entity disabled CLI arg only renames
# the entity to the literal string "disabled", it does not disable wandb).
export WANDB_MODE=disabled

cd /data/wangqiwei/ICLR2026/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-official_libero4in1_qwenpi_0519}

GPUS=${GPUS:-0,2,3,4,5,6}
NUM_PROCESSES=${NUM_PROCESSES:-6}
PORT=${PORT:-29674}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# --- Codex finding 2: resume / overwrite guard ----------------------------
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
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
