#!/bin/bash
# Qwen3.5-0.8B + GR00T + FFS #8 ValueResidual branch.
# Arms:
#   fromscratch: RUN_ID=qwen3p5_0p8b_ffs_value_residual_fromscratch_30k PRETRAINED_CKPT= TRAIN_ONLY=
#   warmstartB:  RUN_ID=qwen3p5_0p8b_ffs_value_residual_warmstartB_30k PRETRAINED_CKPT=<B> TRAIN_ONLY=.attn1.to_v_resid.
set -e

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFS_REPO_DIR=${FFS_REPO_DIR:-/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo}

cd /mnt/data/wangqiwei/wangqiwei/starVLA
export PYTHONPATH=$(pwd):${FFS_REPO_DIR}:${PYTHONPATH:-}

CONDA_VENV=${CONDA_VENV:-/opt/conda/envs/starvla/bin}
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-0.8B}

DATA_ROOT=${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW}
DATA_MIX=${DATA_MIX:-libero_all_sfstereo_leftprimary}
FRAMEWORK=${FRAMEWORK:-QwenGR00T_ValueResidualFFS}
B_CKPT=${B_CKPT:-}
PRETRAINED_CKPT=${PRETRAINED_CKPT-}

ffs_model_path=${FFS_MODEL_PATH:-${FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth}
ffs_sha256=${FFS_SHA256:-98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692}

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-qwen3p5_0p8b_ffs_value_residual_fromscratch_30k}

BS=${BS:-32}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
GPUS=${GPUS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT=${PORT:-29752}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}
FREEZE_MODULES=${FREEZE_MODULES-}
TRAIN_ONLY_DEFAULT=".attn1.to_v_resid."
TRAIN_ONLY=${TRAIN_ONLY-}
CAM_ROPE=${CAM_ROPE:-0}
CAM_BRANCH=${CAM_BRANCH:-0}

case "${CAM_ROPE}" in 0|1) ;; *) echo "[guard] CAM_ROPE must be 0 or 1, got '${CAM_ROPE}'"; exit 3 ;; esac
case "${CAM_BRANCH}" in 0|1) ;; *) echo "[guard] CAM_BRANCH must be 0 or 1, got '${CAM_BRANCH}'"; exit 3 ;; esac
CAM_ROPE_BOOL=$([ "${CAM_ROPE}" = "1" ] && echo true || echo false)
CAM_BRANCH_BOOL=$([ "${CAM_BRANCH}" = "1" ] && echo true || echo false)

[ "${FRAMEWORK}" = "QwenGR00T_ValueResidualFFS" ] || {
  echo "[guard] this launcher requires FRAMEWORK=QwenGR00T_ValueResidualFFS, got '${FRAMEWORK}'"; exit 3;
}
[ "${CAM_ROPE}" = "0" ] || { echo "[guard] ValueResidual arms are defined with CAM_ROPE=0, got '${CAM_ROPE}'"; exit 3; }
[ "${CAM_BRANCH}" = "0" ] || { echo "[guard] ValueResidual arms are defined with CAM_BRANCH=0, got '${CAM_BRANCH}'"; exit 3; }

case "${run_id}" in
  qwen3p5_0p8b_ffs_value_residual_fromscratch_30k)
    [ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] fromscratch requires PRETRAINED_CKPT='', got '${PRETRAINED_CKPT}'"; exit 3; }
    [ -z "${FREEZE_MODULES}" ] || { echo "[guard] fromscratch requires FREEZE_MODULES='', got '${FREEZE_MODULES}'"; exit 3; }
    [ -z "${TRAIN_ONLY}" ] || { echo "[guard] fromscratch trains all params; TRAIN_ONLY must be empty, got '${TRAIN_ONLY}'"; exit 3; }
    ;;
  qwen3p5_0p8b_ffs_value_residual_warmstartB_30k)
    PRETRAINED_CKPT=${PRETRAINED_CKPT:-${B_CKPT}}
    TRAIN_ONLY=${TRAIN_ONLY:-${TRAIN_ONLY_DEFAULT}}
    [ -n "${PRETRAINED_CKPT}" ] || { echo "[guard] warmstartB requires PRETRAINED_CKPT"; exit 3; }
    [ "${TRAIN_ONLY}" = "${TRAIN_ONLY_DEFAULT}" ] || { echo "[guard] warmstartB TRAIN_ONLY must be '${TRAIN_ONLY_DEFAULT}', got '${TRAIN_ONLY}'"; exit 3; }
    [ -z "${FREEZE_MODULES}" ] || { echo "[guard] warmstartB uses TRAIN_ONLY, so FREEZE_MODULES must be empty, got '${FREEZE_MODULES}'"; exit 3; }
    ;;
  *)
    echo "[guard] run_id must be one of the two ValueResidual arms, got '${run_id}'"; exit 3 ;;
esac

preflight_paths=("${ffs_model_path}" "${DS_CONFIG}")
[ -n "${PRETRAINED_CKPT}" ] && preflight_paths+=("${PRETRAINED_CKPT}")
for path in "${preflight_paths[@]}"; do
  [ -f "${path}" ] || { echo "[preflight] missing file: ${path}"; exit 1; }
done

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}/checkpoints
cp "$0" "${output_dir}/" 2>/dev/null || true

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
[ "${#existing_ckpts[@]}" -gt 0 ] && [ "${RESUME:-0}" != "1" ] && { echo "refuse: ckpts exist in ${output_dir}, set RESUME=1 to continue"; exit 1; }
if pgrep -f "run_id ${run_id}\$" > /dev/null 2>&1 || pgrep -f "run_id ${run_id} " > /dev/null 2>&1; then
  echo "refuse: a live process is already training run_id ${run_id}"; exit 1
fi
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true)

if [ -n "${PRETRAINED_CKPT}" ]; then
  echo "[launch] warm-start checkpoint=${PRETRAINED_CKPT}"
  CKPT_FLAG=(--trainer.pretrained_checkpoint "${PRETRAINED_CKPT}")
else
  echo "[launch] from-scratch: no --trainer.pretrained_checkpoint"
  CKPT_FLAG=()
fi

echo "[launch] ${FRAMEWORK} + Qwen3.5-0.8B + GR00T ValueResidual V branch"
echo "[launch] DATA_ROOT=${DATA_ROOT} DATA_MIX=${DATA_MIX}"
echo "[launch] FREEZE_MODULES='${FREEZE_MODULES}' TRAIN_ONLY='${TRAIN_ONLY}'"
echo "[launch] cam_rope=${CAM_ROPE_BOOL} cam_branch=${CAM_BRANCH_BOOL} interleave_self_attention=true"
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
  --framework.action_model.diffusion_model_cfg.interleave_self_attention true \
  --framework.ffs_value_residual.ffs_model_path ${ffs_model_path} \
  --framework.ffs_value_residual.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_value_residual.ffs_feature_source gru_hidden \
  --framework.ffs_value_residual.gru_hidden_dim 16 \
  --framework.ffs_value_residual.ffs_image_size 256 \
  --framework.ffs_value_residual.num_cameras 2 \
  --framework.ffs_value_residual.left_ref_idx 1 \
  --framework.ffs_value_residual.primary_view_idx 0 \
  --framework.ffs_value_residual.inject_cam_id 1 \
  --datasets.vla_data.data_root_dir ${DATA_ROOT} \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  "${CKPT_FLAG[@]}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.train_only "${TRAIN_ONLY}" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval $SAVE_INTERVAL \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Qwen0p8_GR00T_ValueResidual \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
