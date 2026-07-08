#!/bin/bash
# Qwen3.5-0.8B + GR00T, VANILLA BASELINE (no stereo / no FFS / no geometry / no cam_rope / no cam_branch).
# 4-suite reproduction diagnostic (2026-06-28): isolate data-source & camera-config vs the leftprimary
# stereo regression. Runs ONE of:
#   DATA_MIX=libero_all       (primary + wrist)  with DATA_ROOT=LEROBOT_LIBERO_DATA        -> official-pw
#   DATA_MIX=libero_all       (primary + wrist)  with DATA_ROOT=LEROBOT_LIBERO_OURRENDER_PW -> ourrender-pw
#   DATA_MIX=libero_all_mono  (primary only)     with DATA_ROOT=LEROBOT_LIBERO_OURRENDER_PW -> ourrender-mono
# from-scratch full-ft. Built on 4090d (authoritative); runs on a800 (paths below = a800 runtime).
# Geometry is HARDCODED OFF here (this is the vanilla arm) — for stereo use the stereo launcher instead.
set -e

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/wangqiwei/ICLR2026/starVLA                       # a800 runtime path
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

CONDA_VENV=${CONDA_VENV:-/home/wangqiwei/ICLR2026/starVLA/.venv/bin}
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-0.8B}

DATA_ROOT=${DATA_ROOT:?need DATA_ROOT}
DATA_MIX=${DATA_MIX:?need DATA_MIX}
FRAMEWORK=QwenGR00T
PRETRAINED_CKPT=${PRETRAINED_CKPT-}                        # from-scratch by default
FREEZE_MODULES=${FREEZE_MODULES-}                          # full fine-tune by default

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:?need RUN_ID}

# ---- knobs matching the live leftprimary runs (BS16 x ga8 = eff128, 30k) ----
BS=${BS:-16}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
GPUS=${GPUS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT=${PORT:-29760}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga8.yaml}

# ---- vanilla baseline: geometry OFF (hardcoded), non-stereo 4-suite mix only ----
case "${DATA_MIX}" in
  libero_all|libero_all_mono) ;;
  *) echo "[guard] baseline launcher requires DATA_MIX in {libero_all, libero_all_mono}, got '${DATA_MIX}'"; exit 3 ;;
esac
[ "${FRAMEWORK}" = "QwenGR00T" ] || { echo "[guard] requires FRAMEWORK=QwenGR00T"; exit 3; }
# from-scratch / full-ft guard (run_id must declare it)
case "${run_id}" in
  *fromscratch*|*fullft*)
    [ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] run_id '${run_id}' => from-scratch but PRETRAINED_CKPT set. refusing."; exit 3; }
    [ -z "${FREEZE_MODULES}" ]  || { echo "[guard] run_id '${run_id}' => full-ft needs FREEZE_MODULES=''. refusing."; exit 3; } ;;
  *) [ -n "${PRETRAINED_CKPT}" ] || { echo "[guard] non-fromscratch run_id but PRETRAINED_CKPT empty. refusing."; exit 3; } ;;
esac

preflight_paths=("${DS_CONFIG}" "${config_yaml}" "${DATA_ROOT}")
[ -n "${PRETRAINED_CKPT}" ] && preflight_paths+=("${PRETRAINED_CKPT}")
for path in "${preflight_paths[@]}"; do [ -e "${path}" ] || { echo "[preflight] missing: ${path}"; exit 1; }; done

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}/checkpoints
cp "$0" "${output_dir}/" 2>/dev/null || true

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
[ "${#existing_ckpts[@]}" -gt 0 ] && [ "${RESUME:-0}" != "1" ] && { echo "refuse: ckpts exist in ${output_dir}, set RESUME=1"; exit 1; }
if pgrep -f "run_id ${run_id}\$" >/dev/null 2>&1 || pgrep -f "run_id ${run_id} " >/dev/null 2>&1; then
  echo "refuse: a live process is already training run_id ${run_id}"; exit 1; fi
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true)

if [ -n "${PRETRAINED_CKPT}" ]; then CKPT_FLAG=(--trainer.pretrained_checkpoint "${PRETRAINED_CKPT}"); else CKPT_FLAG=(); fi
DS_GA=$(printf '%s' "${DS_CONFIG}" | grep -oE 'ga[0-9]+' | grep -oE '[0-9]+' | tail -1); DS_GA=${DS_GA:-8}
echo "[launch] VANILLA BASELINE QwenGR00T | DATA_MIX=${DATA_MIX} DATA_ROOT=${DATA_ROOT} | BS=$BS x $NUM_PROCESSES x GA${DS_GA} = eff_$((BS*NUM_PROCESSES*DS_GA)) | MAX_STEPS=$MAX_STEPS save=$SAVE_INTERVAL warmup=$WARMUP_STEPS | run_id=$run_id"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${FRAMEWORK} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled false \
  --framework.qwenvl.stereo_cam_branch_enabled false \
  --framework.qwenvl.stereo_epipolar_mask_enabled false \
  --datasets.vla_data.data_root_dir ${DATA_ROOT} \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  "${CKPT_FLAG[@]}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.num_warmup_steps $WARMUP_STEPS \
  --trainer.save_interval $SAVE_INTERVAL \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Qwen0p8_GR00T_Baseline \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
