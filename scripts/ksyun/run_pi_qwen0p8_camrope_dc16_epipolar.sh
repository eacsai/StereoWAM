#!/bin/bash
# Phase 4: PI + Qwen3.5-0.8B + cam_rope d_c=16 + EPIPOLAR mask.
# Baseline: Phase 3 B (PI + 0.8B + cam_rope d_c=16, zero init) = 94% on libero_goal.
# Goal: see if epipolar attention mask (StereoWorld paper Sec 3.3) adds extra +X pp.

set -e
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
run_id=${RUN_ID:-pi_qwen0p8_camrope_dc16_epipolar_$(date +%m%d)}

GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29695}
BS=${BS:-48}
MAX_STEPS=${MAX_STEPS:-30000}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
[ "${#existing_ckpts[@]}" -gt 0 ] && [ "${RESUME:-0}" != "1" ] && { echo refuse; exit 1; }
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true)

echo "[launch] PI + Qwen3.5-0.8B + cam_rope d_c=16 + EPIPOLAR mask"
echo "[launch] BS=$BS × $NUM_PROCESSES GPU = $((BS*NUM_PROCESSES)) eff_batch"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled true \
  --framework.qwenvl.stereo_cam_rope_d_c 16 \
  --framework.qwenvl.stereo_epipolar_mask_enabled true \
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
  --wandb_project starVLA_Epipolar \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
