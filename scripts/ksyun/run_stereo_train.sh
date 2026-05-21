#!/bin/bash
# Stereo VLA training on ksyun H100 dev container (jinshan_dev).
#
# Two modes (toggle via MODE env): MODE=mono | MODE=stereo
#   mono   = Z baseline: video_keys=[primary]              (data_mix=libero_goal_mono_replay)
#   stereo = Y experiment: video_keys=[primary,right_view] (data_mix=libero_goal_stereo)
#
# Both modes use the SAME OpenVLA-faithful dataset rendered by
# scripts/4090d/regenerate_libero_stereo.py (libero_goal_stereo_openvla, 431 episodes).
# The only difference at training time is which video keys the model sees
# — same primary frames, ±right_view — for a fair A/B comparison.
#
# Hardware: 2× H100 80GB on jinshan_dev (10.112.2.128).
# ETA estimate: ~9-12h for 30k steps on 2× H100 (vs 4090d 6× = 12.5h; H100 single
# is ~2x 4090, so 2× H100 ≈ 4× 4090 ≈ 18.75h / 30k * 12.5h = ~6h. Conservative budget 12h).
#
# Adapted from scripts/4090d/run_libero_train_goal_30k.sh — see that for the
# wrist+primary 30k=94% reference run.

set -e

# ── NCCL: single-node 2-GPU, NVLink/PCIe (no IB on container)
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled

# ── Mode selection
MODE=${MODE:-stereo}
if [ "$MODE" = "mono" ]; then
  data_mix=libero_goal_mono_replay
  run_id_default=goal_mono_replay_30k_qwenpi_$(date +%m%d)
elif [ "$MODE" = "stereo" ]; then
  data_mix=libero_goal_stereo
  run_id_default=goal_stereo_30k_qwenpi_$(date +%m%d)
else
  echo "ERROR: MODE must be 'stereo' or 'mono', got: $MODE" >&2
  exit 1
fi

cd /mnt/data/wangqiwei/wangqiwei/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# ── starVLA dataloader autoloads data_mix from each examples/<*>/train_files/data_registry/data_config.py.
# Our libero_goal_stereo / libero_goal_mono_replay mixes live in
# examples/LIBERO_STEREO/train_files/data_registry/data_config.py
# (Libero4in1StereoDataConfig / Libero4in1MonoPrimaryDataConfig).

CONDA_VENV=/opt/conda/envs/starvla/bin
Framework_name=QwenPI
freeze_module_list=''

# TODO: Qwen3.5-0.8B base model path — rsync from 4090d:playground/Pretrained_models/Qwen3.5-0.8B
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B

# TODO: starvla_cotrain_libero.yaml — verify present in container (cloned at c8e98ce should have it).
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml

# TODO: playground/Datasets/LEROBOT_LIBERO_STEREO_DATA → symlink to
#       /mnt/data/wangqiwei/wangqiwei/libero_goal_stereo_openvla (after rsync)
libero_data_root=playground/Datasets/LEROBOT_LIBERO_STEREO_DATA

# TODO: LLaVA-OneVision-COCO vlm cotrain data — rsync from 4090d if missing
#       (default cotrain config expects vlm data; can disable via --datasets.vlm_data... if needed)

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-$run_id_default}

BS=${BS:-48}
MAX_STEPS=${MAX_STEPS:-18000}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29677}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

# ── Resume guard (verbatim from run_libero_train_goal_30k.sh — codex F3 fix)
RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
if [ "${#existing_ckpts[@]}" -gt 0 ]; then
  if [ "${RESUME:-0}" = "1" ]; then
    echo "[resume] ${#existing_ckpts[@]} valid ckpt(s) in ${output_dir}/checkpoints, RESUME=1 → --trainer.is_resume true"
    RESUME_FLAG=(--trainer.is_resume true)
  else
    echo "[refuse] ${output_dir}/checkpoints non-empty. Pick:" >&2
    echo "  RESUME=1 bash $0   |   RUN_ID=${run_id}_v2 bash $0   |   rm -rf ${output_dir}/checkpoints" >&2
    exit 1
  fi
elif [ "${RESUME:-0}" = "1" ]; then
  echo "[refuse] RESUME=1 but no valid steps_*_pytorch_model.pt in ${output_dir}/checkpoints. Aborting." >&2
  exit 1
fi

echo "[launch] MODE=${MODE}  data_mix=${data_mix}  run_id=${run_id}"
echo "[launch] GPUS=${GPUS}  NUM_PROCESSES=${NUM_PROCESSES}"
echo "[launch] data_root=${libero_data_root}  base_vlm=${base_vlm}"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
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
