#!/bin/bash
# Qwen2.5-VL-3B + GR00T on the NEW 4-suite stereo LIBERO data, NO cam_rope.
# Parameterized by DATA_MIX so the same launcher runs all 3 input-view baselines:
#   primary+wrist   -> DATA_MIX=libero_all_mono_selfrender   (robot_type libero_franka)
#   primary only    -> DATA_MIX=libero_all_mono_replay       (libero_franka_mono_primary)
#   primary+right   -> DATA_MIX=libero_all_stereo            (libero_franka_stereo)
# Single-suite smoke variants: libero_spatial_stereo / libero_spatial_mono_replay.
# Recipe (same as the proven upstream Qwen2.5-VL-3B run): BS16 x 2GPU x GA4 = eff_batch 128.
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
base_vlm=${BASE_VLM:-./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct}

# NEW stereo 4-suite data root (symlink dir: libero_{spatial,object,goal,10} -> *_stereo_lerobot)
DATA_ROOT=${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_STEREO_4SUITE}
DATA_MIX=${DATA_MIX:?set DATA_MIX (libero_all_stereo | libero_all_mono_replay | libero_all_mono_selfrender | libero_spatial_stereo ...)}

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:?set RUN_ID}

BS=${BS:-16}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29699}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/" 2>/dev/null || true

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
[ "${#existing_ckpts[@]}" -gt 0 ] && [ "${RESUME:-0}" != "1" ] && { echo "refuse: ckpts exist in ${output_dir}, set RESUME=1 to continue"; exit 1; }
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true)

echo "[launch] QwenGR00T + Qwen2.5-VL-3B-Instruct (NO cam_rope)"
echo "[launch] DATA_ROOT=${DATA_ROOT} DATA_MIX=${DATA_MIX}"
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA4 = eff_128 | MAX_STEPS=$MAX_STEPS | run_id=$run_id"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${FRAMEWORK:-QwenGR00T} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled ${CAM_ROPE:-false} \
  --datasets.vla_data.data_root_dir ${DATA_ROOT} \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval $SAVE_INTERVAL \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Qwen2p5VL_Stereo \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
