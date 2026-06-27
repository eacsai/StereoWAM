#!/bin/bash
# Qwen3.5-0.8B + GR00T + FFS #5 LLaMA-Adapter prefix injection.
set -euo pipefail

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFS_REPO_DIR=${FFS_REPO_DIR:-/home/wangqiwei/ICLR2026/Fast-FoundationStereo}

cd /home/wangqiwei/ICLR2026/starVLA
export PYTHONPATH=$(pwd):${FFS_REPO_DIR}:${PYTHONPATH:-}

CONDA_VENV=${CONDA_VENV:-/home/wangqiwei/ICLR2026/starVLA/.venv/bin}
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-0.8B}

DATA_ROOT=${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW}
DATA_MIX=${DATA_MIX:-libero_all_sfstereo_leftprimary}
FRAMEWORK=QwenGR00T_LlamaAdapterPrefixFFS
B_CKPT=

ffs_model_path=${FFS_MODEL_PATH:-${FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth}
ffs_sha256=${FFS_SHA256:-98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692}

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-qwen3p5_0p8b_ffs_llama_adapter_prefix_warmstartB_frozen_30k}

BS=${BS:-32}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
GPUS=${GPUS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT=${PORT:-29742}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}

CAM_ROPE=${CAM_ROPE:-0}
CAM_BRANCH=${CAM_BRANCH:-0}
case "${CAM_ROPE}" in 0) ;; *) echo "[guard] this FFS #5 launcher forces CAM_ROPE=0, got '${CAM_ROPE}'"; exit 3 ;; esac
case "${CAM_BRANCH}" in 0) ;; *) echo "[guard] this FFS #5 launcher requires CAM_BRANCH=0, got '${CAM_BRANCH}'"; exit 3 ;; esac
CAM_ROPE_BOOL=false
CAM_BRANCH_BOOL=false

case "${run_id}" in
  qwen3p5_0p8b_ffs_llama_adapter_prefix_warmstartB_frozen_30k)
    PRETRAINED_CKPT=${PRETRAINED_CKPT-${B_CKPT}}
    FREEZE_MODULES=${FREEZE_MODULES-qwen_vl_interface}
    [ "${PRETRAINED_CKPT}" = "${B_CKPT}" ] || { echo "[guard] warmstartB_frozen run needs PRETRAINED_CKPT='${B_CKPT}', got '${PRETRAINED_CKPT}'"; exit 3; }
    [ "${FREEZE_MODULES}" = "qwen_vl_interface" ] || { echo "[guard] warmstartB_frozen run needs FREEZE_MODULES=qwen_vl_interface, got '${FREEZE_MODULES}'"; exit 3; }
    ;;
  qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_30k)
    PRETRAINED_CKPT=${PRETRAINED_CKPT-}
    FREEZE_MODULES=${FREEZE_MODULES-}
    [ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] fromscratch run needs empty PRETRAINED_CKPT, got '${PRETRAINED_CKPT}'"; exit 3; }
    [ -z "${FREEZE_MODULES}" ] || { echo "[guard] fromscratch run needs FREEZE_MODULES='', got '${FREEZE_MODULES}'"; exit 3; }
    ;;
  qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_perhead_30k)
    PRETRAINED_CKPT=${PRETRAINED_CKPT-}
    FREEZE_MODULES=${FREEZE_MODULES-}
    [ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] fromscratch perhead run needs empty PRETRAINED_CKPT, got '${PRETRAINED_CKPT}'"; exit 3; }
    [ -z "${FREEZE_MODULES}" ] || { echo "[guard] fromscratch perhead run needs FREEZE_MODULES='', got '${FREEZE_MODULES}'"; exit 3; }
    ;;
  *)
    echo "[guard] unsupported RUN_ID='${run_id}'. Use one of:"
    echo "  qwen3p5_0p8b_ffs_llama_adapter_prefix_warmstartB_frozen_30k"
    echo "  qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_30k"
    echo "  qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_perhead_30k"
    exit 3
    ;;
esac

# per-head gate iff run_id marked _perhead_ (module supports gate_per_head)
case "${run_id}" in *_perhead_*) GATE_PER_HEAD_BOOL=true ;; *) GATE_PER_HEAD_BOOL=false ;; esac

[ -n "${ffs_sha256}" ] || { echo "[guard] FFS_SHA256 / ffs_expected_sha256 must be non-empty"; exit 3; }

preflight_paths=("${ffs_model_path}" "${DS_CONFIG}")
[ -n "${PRETRAINED_CKPT}" ] && preflight_paths+=("${PRETRAINED_CKPT}")
for path in "${preflight_paths[@]}"; do
  [ -f "${path}" ] || { echo "[preflight] missing file: ${path}"; exit 1; }
done
inner_json=$(grep -oE 'deepspeed_config_file: *"?[^"]+' "$DS_CONFIG" | sed -E 's|.*: *"?||' || true)
if [ -n "$inner_json" ] && [ ! -f "$inner_json" ]; then
  echo "[preflight] FATAL: DS_CONFIG wraps inner ${inner_json} which does not exist"; exit 1
fi
echo "[preflight] DS_CONFIG=${DS_CONFIG} inner=${inner_json:-none}"
echo "[preflight] FFS_REPO_DIR=${FFS_REPO_DIR}"
echo "[preflight] ffs_model_path=${ffs_model_path}"
echo "[preflight] ffs_expected_sha256=${ffs_sha256}"

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
echo "[launch] ${FRAMEWORK} + Qwen3.5-0.8B + GR00T + FFS #5 prefix"
echo "[launch] DATA_ROOT=${DATA_ROOT} DATA_MIX=${DATA_MIX}"
echo "[launch] FREEZE_MODULES='${FREEZE_MODULES}'"
echo "[launch] cam_rope: stereo_cam_rope_enabled=${CAM_ROPE_BOOL} (forced off)"
echo "[launch] cam_branch: stereo_cam_branch_enabled=${CAM_BRANCH_BOOL}"
echo "[launch] view indices: left_ref_idx=1 primary_view_idx=0 inject_cam_id=1 num_cameras=2"
echo "[launch] prefix: n_prompts=10 absorb_dim=256 gate_per_head=${GATE_PER_HEAD_BOOL}"
DS_GA=$(printf '%s' "${DS_CONFIG}" | grep -oE 'ga[0-9]+' | grep -oE '[0-9]+' | tail -1 || true); DS_GA=${DS_GA:-4}
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA${DS_GA} (${DS_CONFIG##*/}) = eff_$((BS*NUM_PROCESSES*DS_GA)) | MAX_STEPS=$MAX_STEPS | PORT=$PORT | run_id=$run_id"

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
  --framework.ffs_llama_adapter_prefix.ffs_model_path ${ffs_model_path} \
  --framework.ffs_llama_adapter_prefix.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_llama_adapter_prefix.ffs_feature_source gru_hidden \
  --framework.ffs_llama_adapter_prefix.gru_hidden_dim 16 \
  --framework.ffs_llama_adapter_prefix.ffs_image_size 256 \
  --framework.ffs_llama_adapter_prefix.num_cameras 2 \
  --framework.ffs_llama_adapter_prefix.left_ref_idx 1 \
  --framework.ffs_llama_adapter_prefix.primary_view_idx 0 \
  --framework.ffs_llama_adapter_prefix.inject_cam_id 1 \
  --framework.ffs_llama_adapter_prefix.n_prompts 10 \
  --framework.ffs_llama_adapter_prefix.absorb_dim 256 \
  --framework.ffs_llama_adapter_prefix.gate_per_head ${GATE_PER_HEAD_BOOL} \
  --datasets.vla_data.data_root_dir ${DATA_ROOT} \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  "${CKPT_FLAG[@]}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval $SAVE_INTERVAL \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Qwen0p8_GR00T_FFS5 \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
