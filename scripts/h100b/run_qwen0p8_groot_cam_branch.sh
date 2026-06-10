#!/bin/bash
# Qwen3.5-0.8B + GR00T + parallel PRoPE cam_branch.
set -e

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/data/wangqiwei/wangqiwei/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

CONDA_VENV=${CONDA_VENV:-/opt/conda/envs/starvla/bin}
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-0.8B}

DATA_ROOT=${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW}
DATA_MIX=${DATA_MIX:-libero_all_sfstereo_rightprimary}
FRAMEWORK=QwenGR00T
B_CKPT=playground/Checkpoints/qwen3p5_0p8b_4suite_stereo_camrope_rightprimary_ourrender_30k/checkpoints/steps_30000_pytorch_model.pt
PRETRAINED_CKPT=${PRETRAINED_CKPT-${B_CKPT}}

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-qwen3p5_0p8b_stereo_cam_branch_prope_warmstartB_mechgate_2k}

BS=${BS:-32}
case "${run_id}" in
  *fromscratch*) MAX_STEPS=${MAX_STEPS:-30000} ;;
  *) MAX_STEPS=${MAX_STEPS:-2000} ;;
esac
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
GPUS=${GPUS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT=${PORT:-29741}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}
FREEZE_MODULES=${FREEZE_MODULES-qwen_vl_interface}

CAM_BRANCH=${CAM_BRANCH:-1}
CAM_BRANCH_HEADS=${CAM_BRANCH_HEADS:-4}
CAM_BRANCH_HEAD_DIM=${CAM_BRANCH_HEAD_DIM:-128}
CAM_BRANCH_LAYERS=${CAM_BRANCH_LAYERS:-}
CAM_ROPE=${CAM_ROPE:-0}
case "${CAM_BRANCH}" in 0|1) ;; *) echo "[guard] CAM_BRANCH must be 0 or 1, got '${CAM_BRANCH}'"; exit 3 ;; esac
case "${CAM_ROPE}" in 0|1) ;; *) echo "[guard] CAM_ROPE must be 0 or 1, got '${CAM_ROPE}'"; exit 3 ;; esac
case "${CAM_BRANCH_HEADS}" in ''|*[!0-9]*) echo "[guard] CAM_BRANCH_HEADS must be a positive integer, got '${CAM_BRANCH_HEADS}'"; exit 3 ;; esac
case "${CAM_BRANCH_HEAD_DIM}" in ''|*[!0-9]*) echo "[guard] CAM_BRANCH_HEAD_DIM must be a positive integer, got '${CAM_BRANCH_HEAD_DIM}'"; exit 3 ;; esac
[ "${CAM_BRANCH_HEADS}" -gt 0 ] || { echo "[guard] CAM_BRANCH_HEADS must be > 0"; exit 3; }
{ [ "${CAM_BRANCH_HEAD_DIM}" -gt 0 ] && [ $((CAM_BRANCH_HEAD_DIM % 4)) -eq 0 ]; } || { echo "[guard] CAM_BRANCH_HEAD_DIM must be > 0 and divisible by 4 (PRoPE 4-wide tiles), got '${CAM_BRANCH_HEAD_DIM}'"; exit 3; }
CAM_BRANCH_BOOL=$([ "${CAM_BRANCH}" = "1" ] && echo true || echo false)
CAM_ROPE_BOOL=$([ "${CAM_ROPE}" = "1" ] && echo true || echo false)

# This launcher is for cam_branch experiments only. cam_rope must remain off so the
# old inert dim-expansion patch cannot slow the run or collide with cam_branch.
case "${run_id}" in
  *cam_branch*)
    [ "${CAM_BRANCH}" = "1" ] || { echo "[guard] run_id '${run_id}' => cam_branch but CAM_BRANCH='${CAM_BRANCH}' (need 1). refusing."; exit 3; }
    [ "${CAM_ROPE}" = "0" ] || { echo "[guard] run_id '${run_id}' => cam_branch but CAM_ROPE='${CAM_ROPE}' (need 0). refusing."; exit 3; }
    ;;
esac
case "${run_id}" in
  *mechgate*)
    [ "${PRETRAINED_CKPT}" = "${B_CKPT}" ] || { echo "[guard] run_id '${run_id}' => mechgate but PRETRAINED_CKPT='${PRETRAINED_CKPT}' (need '${B_CKPT}'). refusing."; exit 3; }
    [ "${MAX_STEPS}" = "2000" ] || { echo "[guard] run_id '${run_id}' => mechgate but MAX_STEPS='${MAX_STEPS}' (need 2000). refusing."; exit 3; }
    [ "${FREEZE_MODULES}" = "qwen_vl_interface" ] || { echo "[guard] run_id '${run_id}' => mechgate needs FREEZE_MODULES=qwen_vl_interface (frozen VLM is the defining gate property) but got '${FREEZE_MODULES}'. refusing."; exit 3; }
    ;;
esac

# The yaml's num_warmup_steps (5000) exceeds the whole 2000-step mechgate: without an
# override the branch LR never leaves the warmup ramp (peaks at 40% of base), weakening
# the "branch learns" readout ~5x. Short gate gets a short warmup; 30k arms keep the yaml.
WARMUP_FLAG=()
case "${run_id}" in
  *mechgate*) WARMUP_FLAG=(--trainer.num_warmup_steps 200) ;;
esac
case "${run_id}" in
  *fromscratch*)
    [ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] run_id '${run_id}' => fromscratch but PRETRAINED_CKPT='${PRETRAINED_CKPT}' is set. refusing."; exit 3; }
    [ -z "${FREEZE_MODULES}" ] || { echo "[guard] run_id '${run_id}' => fromscratch needs FREEZE_MODULES='' (full fine-tune) but got '${FREEZE_MODULES}'. refusing."; exit 3; }
    ;;
  *)
    # A non-fromscratch run with an empty checkpoint would silently train from
    # scratch under a warm-start-looking run_id.
    [ -n "${PRETRAINED_CKPT}" ] || { echo "[guard] run_id '${run_id}' is not *fromscratch* but PRETRAINED_CKPT is empty — refusing a silent from-scratch run."; exit 3; }
    ;;
esac

preflight_paths=("${DS_CONFIG}")
[ -n "${PRETRAINED_CKPT}" ] && preflight_paths+=("${PRETRAINED_CKPT}")
for path in "${preflight_paths[@]}"; do
  [ -f "${path}" ] || { echo "[preflight] missing file: ${path}"; exit 1; }
done

EXTRA_BRANCH_FLAGS=()
if [ -n "${CAM_BRANCH_LAYERS}" ]; then
  EXTRA_BRANCH_FLAGS=(--framework.qwenvl.stereo_cam_branch_layers "${CAM_BRANCH_LAYERS}")
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}/checkpoints
cp "$0" "${output_dir}/" 2>/dev/null || true

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
[ "${#existing_ckpts[@]}" -gt 0 ] && [ "${RESUME:-0}" != "1" ] && { echo "refuse: ckpts exist in ${output_dir}, set RESUME=1 to continue"; exit 1; }
# A live run before its first checkpoint save has an output dir but no steps_* file —
# refuse while any process is already training this run_id (duplicate-launch guard).
if pgrep -f "run_id ${run_id}\$" > /dev/null 2>&1 || pgrep -f "run_id ${run_id} " > /dev/null 2>&1; then
  echo "refuse: a live process is already training run_id ${run_id}"; exit 1
fi
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true)

echo "[launch] ${FRAMEWORK} + Qwen3.5-0.8B + GR00T + cam_branch parallel PRoPE"
if [ -n "${PRETRAINED_CKPT}" ]; then
  echo "[launch] warm-start checkpoint=${PRETRAINED_CKPT}"
  CKPT_FLAG=(--trainer.pretrained_checkpoint "${PRETRAINED_CKPT}")
else
  echo "[launch] from-scratch: no --trainer.pretrained_checkpoint"
  CKPT_FLAG=()
fi
echo "[launch] DATA_ROOT=${DATA_ROOT} DATA_MIX=${DATA_MIX}"
echo "[launch] FREEZE_MODULES='${FREEZE_MODULES}'"
echo "[launch] cam_rope: stereo_cam_rope_enabled=${CAM_ROPE_BOOL} (forced off for cam_branch)"
echo "[launch] cam_branch: enabled=${CAM_BRANCH_BOOL} heads=${CAM_BRANCH_HEADS} head_dim=${CAM_BRANCH_HEAD_DIM} layers='${CAM_BRANCH_LAYERS:-default}'"
DS_GA=$(printf '%s' "${DS_CONFIG}" | grep -oE 'ga[0-9]+' | grep -oE '[0-9]+' | tail -1); DS_GA=${DS_GA:-4}
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA${DS_GA} (${DS_CONFIG##*/}) = eff_$((BS*NUM_PROCESSES*DS_GA)) | MAX_STEPS=$MAX_STEPS | run_id=$run_id"

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
  "${EXTRA_BRANCH_FLAGS[@]}" \
  --datasets.vla_data.data_root_dir ${DATA_ROOT} \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  "${CKPT_FLAG[@]}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.max_train_steps $MAX_STEPS \
  "${WARMUP_FLAG[@]}" \
  --trainer.save_interval $SAVE_INTERVAL \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Qwen0p8_GR00T_CamBranch \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
