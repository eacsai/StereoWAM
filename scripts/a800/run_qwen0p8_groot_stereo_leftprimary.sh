#!/bin/bash
# Qwen3.5-0.8B + GR00T, PURE STEREO (no FFS / no scene-flow / no depth-token / no utonia).
# leftprimary convention, from-scratch full-ft. Geometry arm selectable:
#   CAM_BRANCH=1  -> StereoWorld parallel-PRoPE branch (real geometry-on, it LEARNS; single-side
#                    zero-init out_proj per cam_branch_attention.py:226-238)
#   CAM_ROPE=1    -> legacy d_c=16 cam_rope (KNOWN INERT double-zero no-op + ~4x slower; do NOT use
#                    as the geometry arm — it never learns. Kept only for a literal cam_rope ablation.)
# Built on 4090d (authoritative); runs on a800 (paths below are a800 runtime). Sync 4090d->a800 to run.
set -e

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/wangqiwei/ICLR2026/starVLA                       # a800 runtime path (matches ffs.sh:36)
export PYTHONPATH=$(pwd):${PYTHONPATH:-}                   # no FFS_REPO_DIR needed for pure stereo

CONDA_VENV=${CONDA_VENV:-/home/wangqiwei/ICLR2026/starVLA/.venv/bin}
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-0.8B}

DATA_ROOT=${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW}
DATA_MIX=${DATA_MIX:-libero_all_sfstereo_leftprimary}
FRAMEWORK=QwenGR00T
PRETRAINED_CKPT=${PRETRAINED_CKPT-}                        # from-scratch by default
FREEZE_MODULES=${FREEZE_MODULES-}                          # full fine-tune by default

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-qwen0p8_groot_stereo_plain_leftprimary_fullft_30k}

# ---- knobs matching the 3 live leftprimary runs (BS16 x ga8 = eff128, 30k, save2000) ----
BS=${BS:-16}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
GPUS=${GPUS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT=${PORT:-29770}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga8.yaml}

# ---- geometry arm ----
CAM_ROPE=${CAM_ROPE:-0}
CAM_BRANCH=${CAM_BRANCH:-0}
CAM_BRANCH_HEADS=${CAM_BRANCH_HEADS:-4}
CAM_BRANCH_HEAD_DIM=${CAM_BRANCH_HEAD_DIM:-128}
case "${CAM_ROPE}"   in 0|1) ;; *) echo "[guard] CAM_ROPE must be 0 or 1, got '${CAM_ROPE}'"; exit 3 ;; esac
case "${CAM_BRANCH}" in 0|1) ;; *) echo "[guard] CAM_BRANCH must be 0 or 1, got '${CAM_BRANCH}'"; exit 3 ;; esac
[ "${CAM_ROPE}" = "1" ] && [ "${CAM_BRANCH}" = "1" ] && { echo "[guard] CAM_ROPE and CAM_BRANCH are mutually exclusive"; exit 3; }
CAM_ROPE_BOOL=$([ "${CAM_ROPE}" = "1" ] && echo true || echo false)
CAM_BRANCH_BOOL=$([ "${CAM_BRANCH}" = "1" ] && echo true || echo false)

# ---- fail-closed guards (mirror ffs.sh / cam_branch.sh style) ----
[ "${FRAMEWORK}" = "QwenGR00T" ] || { echo "[guard] pure stereo requires FRAMEWORK=QwenGR00T"; exit 3; }
[ "${DATA_MIX}" = "libero_all_sfstereo_leftprimary" ] || { echo "[guard] requires DATA_MIX=libero_all_sfstereo_leftprimary, got '${DATA_MIX}'"; exit 3; }
case "${run_id}" in
  *fromscratch*|*fullft*)
    [ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] run_id '${run_id}' => from-scratch but PRETRAINED_CKPT set. refusing."; exit 3; }
    [ -z "${FREEZE_MODULES}" ]  || { echo "[guard] run_id '${run_id}' => full-ft needs FREEZE_MODULES=''. refusing."; exit 3; } ;;
  *) [ -n "${PRETRAINED_CKPT}" ] || { echo "[guard] non-fromscratch run_id but PRETRAINED_CKPT empty. refusing."; exit 3; } ;;
esac
case "${run_id}" in
  *cambranch*) [ "${CAM_BRANCH}" = "1" ] || { echo "[guard] run_id says cambranch but CAM_BRANCH!=1"; exit 3; } ;;
  *camrope*)   [ "${CAM_ROPE}"   = "1" ] || { echo "[guard] run_id says camrope but CAM_ROPE!=1"; exit 3; } ;;
  *stereo_plain*) { [ "${CAM_ROPE}" = "0" ] && [ "${CAM_BRANCH}" = "0" ]; } || { echo "[guard] plain arm needs CAM_ROPE=0 CAM_BRANCH=0"; exit 3; } ;;
esac

preflight_paths=("${DS_CONFIG}" "${config_yaml}")
[ -n "${PRETRAINED_CKPT}" ] && preflight_paths+=("${PRETRAINED_CKPT}")
for path in "${preflight_paths[@]}"; do [ -f "${path}" ] || { echo "[preflight] missing file: ${path}"; exit 1; }; done

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
DS_GA=$(printf '%s' "${DS_CONFIG}" | grep -oE 'ga[0-9]+' | grep -oE '[0-9]+' | tail -1); DS_GA=${DS_GA:-4}
echo "[launch] PURE STEREO QwenGR00T | cam_rope=${CAM_ROPE_BOOL} cam_branch=${CAM_BRANCH_BOOL} | BS=$BS x $NUM_PROCESSES x GA${DS_GA} = eff_$((BS*NUM_PROCESSES*DS_GA)) | MAX_STEPS=$MAX_STEPS save=$SAVE_INTERVAL warmup=$WARMUP_STEPS | run_id=$run_id"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${FRAMEWORK} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled ${CAM_ROPE_BOOL} \
  --framework.qwenvl.stereo_cam_rope_d_c 16 \
  --framework.qwenvl.stereo_cam_rope_num_cameras 2 \
  --framework.qwenvl.stereo_cam_rope_baseline_m 0.06 \
  --framework.qwenvl.stereo_cam_rope_fovy_degrees 45.0 \
  --framework.qwenvl.stereo_cam_rope_image_width 256 \
  --framework.qwenvl.stereo_cam_rope_image_height 256 \
  --framework.qwenvl.stereo_cam_rope_spatial_merge 2 \
  --framework.qwenvl.stereo_cam_rope_init_mode zero \
  --framework.qwenvl.stereo_cam_rope_right_first true \
  --framework.qwenvl.stereo_epipolar_mask_enabled false \
  --framework.qwenvl.stereo_cam_branch_enabled ${CAM_BRANCH_BOOL} \
  --framework.qwenvl.stereo_cam_branch_heads ${CAM_BRANCH_HEADS} \
  --framework.qwenvl.stereo_cam_branch_head_dim ${CAM_BRANCH_HEAD_DIM} \
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
  --wandb_project starVLA_Qwen0p8_GR00T_Stereo \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
