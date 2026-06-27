#!/bin/bash
# Qwen3.5-0.8B + GR00T + CamRoPE + FFS methods #1/#2/#3/#4.
# Select method with:
#   FRAMEWORK=QwenGR00T_VLMInputFFS
#   FRAMEWORK=QwenGR00T_ControlNetFFS
#   FRAMEWORK=QwenGR00T_DepthTokenFFS
#   FRAMEWORK=QwenGR00T_DepthImageFFS
set -e

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFS_REPO_DIR=${FFS_REPO_DIR:-/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo}

cd /data/wangqiwei/ICLR2026/starVLA
export PYTHONPATH=$(pwd):${FFS_REPO_DIR}:${PYTHONPATH:-}

CONDA_VENV=${CONDA_VENV:-/opt/conda/envs/starvla/bin}
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-0.8B}

DATA_ROOT=${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW}
DATA_MIX=${DATA_MIX:-libero_all_sfstereo_leftprimary}
FRAMEWORK=${FRAMEWORK:-QwenGR00T_VLMInputFFS}
PRETRAINED_CKPT=${PRETRAINED_CKPT-}

ffs_model_path=${FFS_MODEL_PATH:-${FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth}
ffs_sha256=${FFS_SHA256:-98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692}

case "${FRAMEWORK}" in
  QwenGR00T_VLMInputFFS) DEFAULT_GPUS=0; DEFAULT_PORT=29731 ;;
  QwenGR00T_ControlNetFFS) DEFAULT_GPUS=1; DEFAULT_PORT=29732 ;;
  QwenGR00T_DepthTokenFFS) DEFAULT_GPUS=0; DEFAULT_PORT=29734 ;;
  QwenGR00T_DepthImageFFS) DEFAULT_GPUS=0; DEFAULT_PORT=29735 ;;
  *) echo "FRAMEWORK must be one of QwenGR00T_VLMInputFFS / QwenGR00T_ControlNetFFS / QwenGR00T_DepthTokenFFS / QwenGR00T_DepthImageFFS"; exit 1 ;;
esac

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-qwen0p8_groot_ffs_${FRAMEWORK}_$(date +%m%d)}

BS=${BS:-32}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
GPUS=${GPUS:-${DEFAULT_GPUS}}
NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT=${PORT:-${DEFAULT_PORT}}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}
FREEZE_MODULES=${FREEZE_MODULES-qwen_vl_interface}
INJECT_GATE_INIT=${INJECT_GATE_INIT:-zero}
# Depth-token action-head-access ablation knobs. Defaults reproduce the existing
# behavior byte-for-byte (16 tokens / 4x4 pool / keep). New keep/strip runs pass
# NUM_DEPTH_TOKENS=64 POOL_HW=8 and STRIP_DEPTH=0/1.
STRIP_DEPTH=${STRIP_DEPTH:-0}
NUM_DEPTH_TOKENS=${NUM_DEPTH_TOKENS:-16}
POOL_HW=${POOL_HW:-4}
FFS_CACHE_DIR=${FFS_CACHE_DIR:-}
case "${STRIP_DEPTH}" in 0|1) ;; *) echo "[guard] STRIP_DEPTH must be 0 or 1, got '${STRIP_DEPTH}'"; exit 3 ;; esac
# CAM_ROPE=0 skips installing the stereo cam_rope attention patch. The patch is provably
# inert (q_cam_proj AND k_cam_proj zero-init -> bilinear output and grads identically 0)
# yet its d_c=16 branch widens head_dim 256->272 past FlashAttention2's hard limit,
# forcing the SDPA math backend -> ~4x slower full-finetune steps. Disabling it is
# mathematically output-equivalent (verified by smoke_camrope_disable_equivalence.py)
# and restores FA2. Default 1 keeps every pre-existing run type byte-identical.
CAM_ROPE=${CAM_ROPE:-0}
case "${CAM_ROPE}" in 0|1) ;; *) echo "[guard] CAM_ROPE must be 0 or 1, got '${CAM_ROPE}'"; exit 3 ;; esac
CAM_ROPE_BOOL=$([ "${CAM_ROPE}" = "1" ] && echo true || echo false)

# Dense scene-flow auxiliary supervision is default-off. When enabled, the model
# instantiates action_model.scene_flow_decoder and the dataloader reads sidecars.
SCENE_FLOW=${SCENE_FLOW:-0}
case "${SCENE_FLOW}" in 0|1) ;; *) echo "[guard] SCENE_FLOW must be 0 or 1, got '${SCENE_FLOW}'"; exit 3 ;; esac
SCENE_FLOW_BOOL=$([ "${SCENE_FLOW}" = "1" ] && echo true || echo false)
FLOW_LAMBDA=${FLOW_LAMBDA:-0.05}
FLOW_GRID=${FLOW_GRID:-16}
FLOW_HIDDEN_LAYER=${FLOW_HIDDEN_LAYER:--1}
FLOW_MASK=${FLOW_MASK:-dynamic}
FLOW_GT_ONLY=${FLOW_GT_ONLY:-1}
case "${FLOW_GT_ONLY}" in 0|1) ;; *) echo "[guard] FLOW_GT_ONLY must be 0 or 1, got '${FLOW_GT_ONLY}'"; exit 3 ;; esac
FLOW_GT_ONLY_BOOL=$([ "${FLOW_GT_ONLY}" = "1" ] && echo true || echo false)
FLOW_GT_DIR=${FLOW_GT_DIR:-}
FLOW_INDEX=${FLOW_INDEX:-}
FLOW_EXPECTED_FLIP=${FLOW_EXPECTED_FLIP:-rot180}
FLOW_STEP0_WARMUP=${FLOW_STEP0_WARMUP:-1}
FLOW_STEP0_AUDIT=${FLOW_STEP0_AUDIT:-1}
case "${FLOW_STEP0_AUDIT}" in 0|1) ;; *) echo "[guard] FLOW_STEP0_AUDIT must be 0 or 1, got '${FLOW_STEP0_AUDIT}'"; exit 3 ;; esac
FLOW_STEP0_AUDIT_BOOL=$([ "${FLOW_STEP0_AUDIT}" = "1" ] && echo true || echo false)
FLOW_GRAD_RATIO_STEPS=${FLOW_GRAD_RATIO_STEPS:-20}
FLOW_ONLINE_GRADNORM=${FLOW_ONLINE_GRADNORM:-0}
case "${FLOW_ONLINE_GRADNORM}" in 0|1) ;; *) echo "[guard] FLOW_ONLINE_GRADNORM must be 0 or 1, got '${FLOW_ONLINE_GRADNORM}'"; exit 3 ;; esac
[ "${SCENE_FLOW}" = "0" ] && [ "${FLOW_ONLINE_GRADNORM}" = "1" ] && { echo "[guard] FLOW_ONLINE_GRADNORM=1 requires SCENE_FLOW=1"; exit 3; }
FLOW_ONLINE_GRADNORM_BOOL=$([ "${FLOW_ONLINE_GRADNORM}" = "1" ] && echo true || echo false)
FLOW_TARGET_GRAD_RATIO=${FLOW_TARGET_GRAD_RATIO:-0.05}
FLOW_LAMBDA_MIN=${FLOW_LAMBDA_MIN:-0.001}
FLOW_LAMBDA_MAX=${FLOW_LAMBDA_MAX:-10000}
FLOW_LAMBDA_EMA_DECAY=${FLOW_LAMBDA_EMA_DECAY:-0.97}
FLOW_GRADNORM_EVERY_N=${FLOW_GRADNORM_EVERY_N:-10}
FLOW_GRADNORM_PROBE_SAMPLES=${FLOW_GRADNORM_PROBE_SAMPLES:-4}
FLOW_LAMBDA_JUMP_CAP=${FLOW_LAMBDA_JUMP_CAP:-2.0}
FLOW_LAMBDA_WARMUP_STEPS=${FLOW_LAMBDA_WARMUP_STEPS:-100}
FLOW_MIN_SUPERVISED_PIXELS=${FLOW_MIN_SUPERVISED_PIXELS:-128}

SCENE_FLOW_FLAGS=()
if [ "${SCENE_FLOW}" = "1" ]; then
  "${CONDA_VENV}/python" - \
    "${FLOW_TARGET_GRAD_RATIO}" \
    "${FLOW_LAMBDA_MIN}" \
    "${FLOW_LAMBDA_MAX}" \
    "${FLOW_LAMBDA_EMA_DECAY}" \
    "${FLOW_GRADNORM_EVERY_N}" \
    "${FLOW_GRADNORM_PROBE_SAMPLES}" \
    "${FLOW_LAMBDA_JUMP_CAP}" \
    "${FLOW_LAMBDA_WARMUP_STEPS}" \
    "${FLOW_MIN_SUPERVISED_PIXELS}" <<'PY'
import sys

target, lambda_min, lambda_max, decay, every_n, probe_samples, jump_cap, warmup_steps, min_pixels = sys.argv[1:]
target = float(target)
lambda_min = float(lambda_min)
lambda_max = float(lambda_max)
decay = float(decay)
every_n = int(every_n)
probe_samples = int(probe_samples)
jump_cap = float(jump_cap)
warmup_steps = int(warmup_steps)
min_pixels = float(min_pixels)

if target <= 0:
    raise SystemExit(f"[guard] FLOW_TARGET_GRAD_RATIO must be > 0, got {target}")
if lambda_min > lambda_max:
    raise SystemExit(f"[guard] FLOW_LAMBDA_MIN must be <= FLOW_LAMBDA_MAX, got {lambda_min} > {lambda_max}")
if not (0.95 <= decay <= 0.98):
    raise SystemExit(f"[guard] FLOW_LAMBDA_EMA_DECAY must be in [0.95, 0.98], got {decay}")
if every_n <= 0:
    raise SystemExit(f"[guard] FLOW_GRADNORM_EVERY_N must be > 0, got {every_n}")
if not (2 <= probe_samples <= 4):
    raise SystemExit(f"[guard] FLOW_GRADNORM_PROBE_SAMPLES must be in [2, 4], got {probe_samples}")
if not (1.0 <= jump_cap <= 2.0):
    raise SystemExit(f"[guard] FLOW_LAMBDA_JUMP_CAP must be in [1, 2], got {jump_cap}")
if warmup_steps < 0:
    raise SystemExit(f"[guard] FLOW_LAMBDA_WARMUP_STEPS must be >= 0, got {warmup_steps}")
if min_pixels < 0:
    raise SystemExit(f"[guard] FLOW_MIN_SUPERVISED_PIXELS must be >= 0, got {min_pixels}")
PY
  SCENE_FLOW_FLAGS+=(
    --framework.action_model.scene_flow.enabled true
    --framework.action_model.scene_flow.flow_lambda "${FLOW_LAMBDA}"
    --framework.action_model.scene_flow.online_grad_norm_enabled "${FLOW_ONLINE_GRADNORM_BOOL}"
    --framework.action_model.scene_flow.target_grad_ratio "${FLOW_TARGET_GRAD_RATIO}"
    --framework.action_model.scene_flow.lambda_min "${FLOW_LAMBDA_MIN}"
    --framework.action_model.scene_flow.lambda_max "${FLOW_LAMBDA_MAX}"
    --framework.action_model.scene_flow.lambda_ema_decay "${FLOW_LAMBDA_EMA_DECAY}"
    --framework.action_model.scene_flow.online_grad_norm_every_n_steps "${FLOW_GRADNORM_EVERY_N}"
    --framework.action_model.scene_flow.probe_samples "${FLOW_GRADNORM_PROBE_SAMPLES}"
    --framework.action_model.scene_flow.lambda_jump_cap "${FLOW_LAMBDA_JUMP_CAP}"
    --framework.action_model.scene_flow.lambda_warmup_steps "${FLOW_LAMBDA_WARMUP_STEPS}"
    --framework.action_model.scene_flow.min_supervised_pixels "${FLOW_MIN_SUPERVISED_PIXELS}"
    --framework.action_model.scene_flow.grid_size "${FLOW_GRID}"
    --framework.action_model.scene_flow.hidden_layer "${FLOW_HIDDEN_LAYER}"
    --framework.action_model.scene_flow.mask_mode "${FLOW_MASK}"
    --framework.action_model.scene_flow.step0_flow_warmup_steps "${FLOW_STEP0_WARMUP}"
    --framework.action_model.scene_flow.step0_action_loss_audit "${FLOW_STEP0_AUDIT_BOOL}"
    --framework.action_model.scene_flow.grad_ratio_steps "${FLOW_GRAD_RATIO_STEPS}"
    --datasets.vla_data.scene_flow.enabled true
    --datasets.vla_data.scene_flow.gt_only_sampler "${FLOW_GT_ONLY_BOOL}"
    --datasets.vla_data.scene_flow.expected_sidecar_to_training_flip "${FLOW_EXPECTED_FLIP}"
  )
  [ -n "${FLOW_GT_DIR}" ] && SCENE_FLOW_FLAGS+=(--datasets.vla_data.scene_flow.gt_dir "${FLOW_GT_DIR}")
  [ -n "${FLOW_INDEX}" ] && SCENE_FLOW_FLAGS+=(--datasets.vla_data.scene_flow.index_path "${FLOW_INDEX}")
fi

FFS_CACHE_FLAGS=()
if [ -n "${FFS_CACHE_DIR}" ]; then
  FFS_CACHE_FLAGS=(--framework.ffs_depth_token.ffs_cache_dir "${FFS_CACHE_DIR}")
fi

EXTRA_QWENVL_FLAGS=()
if [ "${FRAMEWORK}" = "QwenGR00T_DepthImageFFS" ]; then
  EXTRA_IMAGE_CAM_ID=${EXTRA_IMAGE_CAM_ID:-1}
  EXTRA_QWENVL_FLAGS=(--framework.qwenvl.stereo_extra_image_cam_id "${EXTRA_IMAGE_CAM_ID}")
fi

# Fail-closed guards: the run_id encodes experiment intent, so a missing/incorrect
# env override cannot silently run the wrong configuration.
case "${run_id}" in
  *fullinject*)
    [ "${INJECT_GATE_INIT}" = "identity" ] || { echo "[guard] run_id '${run_id}' => full injection but INJECT_GATE_INIT='${INJECT_GATE_INIT}' (need 'identity'). refusing."; exit 3; }
    # INJECT_GATE_INIT is only consumed by ffs_vlm_input (framework #1); other
    # frameworks silently ignore it, so a fullinject run_id with the wrong FRAMEWORK
    # would pass the env guard yet train a zero-gate run under a fullinject label.
    [ "${FRAMEWORK}" = "QwenGR00T_VLMInputFFS" ] || { echo "[guard] run_id '${run_id}' => fullinject requires FRAMEWORK=QwenGR00T_VLMInputFFS (the only consumer of INJECT_GATE_INIT) but got '${FRAMEWORK}'. refusing."; exit 3; }
    ;;
esac
case "${run_id}" in
  *fromscratch*) [ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] run_id '${run_id}' => from-scratch but PRETRAINED_CKPT='${PRETRAINED_CKPT}' is set. refusing."; exit 3; } ;;
  *warmstart*) [ -n "${PRETRAINED_CKPT}" ] || { echo "[guard] run_id '${run_id}' => warm-start but PRETRAINED_CKPT is empty — refusing a silent from-scratch run."; exit 3; } ;;
esac
# from-scratch / full-injection runs MUST be full fine-tune (FREEZE_MODULES="") to
# stay a fair single-variable comparison vs the full-finetune B baseline. The launcher
# default is qwen_vl_interface, so without an explicit FREEZE_MODULES="" override this
# would silently train a frozen-VLM run -> wasted GPU-days on an unfair result.
case "${run_id}" in
  *fromscratch*|*fullinject*) [ -z "${FREEZE_MODULES}" ] || { echo "[guard] run_id '${run_id}' => from-scratch/full-injection needs FREEZE_MODULES='' (full fine-tune, fair vs B) but got '${FREEZE_MODULES}'. refusing."; exit 3; } ;;
esac
# Depth-token action-head-access ablation: run_id pins the strip flag so a keep run
# cannot silently strip (or vice-versa).
case "${run_id}" in
  *depthtoken_keep*)  [ "${STRIP_DEPTH}" = "0" ] || { echo "[guard] run_id '${run_id}' => keep but STRIP_DEPTH='${STRIP_DEPTH}' (need 0). refusing."; exit 3; } ;;
  *depthtoken_strip*) [ "${STRIP_DEPTH}" = "1" ] || { echo "[guard] run_id '${run_id}' => strip but STRIP_DEPTH='${STRIP_DEPTH}' (need 1). refusing."; exit 3; } ;;
esac
STRIP_DEPTH_BOOL=$([ "${STRIP_DEPTH}" = "1" ] && echo true || echo false)

# The #4 from-scratch arms are DEFINED as cam_rope-off runs (inert-bypass, FA2 fast
# path); a manual relaunch that forgets CAM_ROPE=0 must not silently train the 4x
# slower cam_rope-on configuration under the same run_id.
case "${run_id}" in
  *depthtoken_keep*|*depthtoken_strip*|*depthimage*)
    case "${run_id}" in
      *fromscratch*) [ "${CAM_ROPE}" = "0" ] || { echo "[guard] run_id '${run_id}' => #4 cam_rope-off arm but CAM_ROPE='${CAM_ROPE}' (need 0). refusing."; exit 3; } ;;
    esac
    ;;
esac

# Fail-closed (codex Stage-4 HIGH): the run_id encodes the framework + token-count intent,
# so a forgotten FRAMEWORK / token override must not silently launch a mislabeled run that
# is later evaluated by folder name.
case "${run_id}" in
  *depthtoken_keep*|*depthtoken_strip*)
    [ "${FRAMEWORK}" = "QwenGR00T_DepthTokenFFS" ] || { echo "[guard] run_id '${run_id}' => depth-token but FRAMEWORK='${FRAMEWORK}' (need QwenGR00T_DepthTokenFFS). refusing."; exit 3; }
    { [ "${NUM_DEPTH_TOKENS}" = "64" ] && [ "${POOL_HW}" = "8" ]; } || { echo "[guard] run_id '${run_id}' => depth-token ablation needs NUM_DEPTH_TOKENS=64 POOL_HW=8 but got '${NUM_DEPTH_TOKENS}'/'${POOL_HW}'. refusing."; exit 3; }
    ;;
  *depthimage*)
    [ "${FRAMEWORK}" = "QwenGR00T_DepthImageFFS" ] || { echo "[guard] run_id '${run_id}' => depth-image but FRAMEWORK='${FRAMEWORK}' (need QwenGR00T_DepthImageFFS). refusing."; exit 3; }
    ;;
esac

preflight_paths=("${ffs_model_path}" "${DS_CONFIG}")
[ -n "${PRETRAINED_CKPT}" ] && preflight_paths+=("${PRETRAINED_CKPT}")
[ "${SCENE_FLOW}" = "1" ] && [ -n "${FLOW_INDEX}" ] && preflight_paths+=("${FLOW_INDEX}")
for path in "${preflight_paths[@]}"; do
  [ -f "${path}" ] || { echo "[preflight] missing file: ${path}"; exit 1; }
done
if [ "${SCENE_FLOW}" = "1" ] && [ -n "${FLOW_GT_DIR}" ] && [ ! -d "${FLOW_GT_DIR}" ]; then
  echo "[preflight] missing scene-flow GT dir: ${FLOW_GT_DIR}"
  exit 1
fi
if [ -n "${FFS_CACHE_DIR}" ] && [ ! -d "${FFS_CACHE_DIR}" ]; then
  echo "[preflight] missing FFS net0 cache dir: ${FFS_CACHE_DIR}"
  exit 1
fi
if [ "${SCENE_FLOW}" = "1" ] && [ -n "${FLOW_INDEX}" ]; then
  "${CONDA_VENV}/python" - "${FLOW_INDEX}" "${FLOW_EXPECTED_FLIP}" <<'PY'
import json
import sys

index_path, expected = sys.argv[1], sys.argv[2].lower()
with open(index_path, "r") as f:
    data = json.load(f)
audit = data.get("alignment_audit") if isinstance(data, dict) else None
if not isinstance(audit, dict) or audit.get("enabled") is not True or not audit.get("global_flip"):
    raise SystemExit(
        f"[preflight] scene-flow index must contain alignment_audit.enabled=true and global_flip: {index_path}"
    )
global_flip = str(audit["global_flip"]).lower()
if expected not in {"", "any", "none"} and global_flip != expected:
    raise SystemExit(
        f"[preflight] scene-flow index global_flip={global_flip!r}, expected {expected!r}: {index_path}"
    )
print(f"[preflight] scene_flow alignment_audit global_flip={global_flip}")
PY
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
# the guard above would not stop a duplicate launch from corrupting it. Refuse while
# any process is already training this run_id.
if pgrep -f "run_id ${run_id}\$" > /dev/null 2>&1 || pgrep -f "run_id ${run_id} " > /dev/null 2>&1; then
  echo "refuse: a live process is already training run_id ${run_id}"; exit 1
fi
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true)

echo "[launch] ${FRAMEWORK} + Qwen3.5-0.8B + GR00T + CamRoPE right-first"
if [ -n "${PRETRAINED_CKPT}" ]; then
  echo "[launch] warm-start checkpoint=${PRETRAINED_CKPT}"
  CKPT_FLAG=(--trainer.pretrained_checkpoint "${PRETRAINED_CKPT}")
else
  echo "[launch] from-scratch: no --trainer.pretrained_checkpoint"
  CKPT_FLAG=()
fi
echo "[launch] DATA_ROOT=${DATA_ROOT} DATA_MIX=${DATA_MIX}"
echo "[launch] FREEZE_MODULES='${FREEZE_MODULES}'"
echo "[launch] INJECT_GATE_INIT='${INJECT_GATE_INIT}'"
echo "[launch] depth-token: num_depth_tokens=${NUM_DEPTH_TOKENS} pool_hw=${POOL_HW} strip_depth_tokens=${STRIP_DEPTH_BOOL}"
[ -n "${FFS_CACHE_DIR}" ] && echo "[launch] depth-token FFS net0 cache: ${FFS_CACHE_DIR}"
echo "[launch] cam_rope: stereo_cam_rope_enabled=${CAM_ROPE_BOOL} (CAM_ROPE=${CAM_ROPE}; 0 = bypass inert cam_rope, keep FlashAttention)"
if [ "${SCENE_FLOW}" = "1" ]; then
  echo "[launch] scene_flow: lambda=${FLOW_LAMBDA} grid=${FLOW_GRID} hidden_layer=${FLOW_HIDDEN_LAYER} mask=${FLOW_MASK} gt_only=${FLOW_GT_ONLY_BOOL} step0_audit=${FLOW_STEP0_AUDIT_BOOL}"
  echo "[launch] scene_flow online_gradnorm: enabled=${FLOW_ONLINE_GRADNORM_BOOL} target=${FLOW_TARGET_GRAD_RATIO} lambda_range=[${FLOW_LAMBDA_MIN},${FLOW_LAMBDA_MAX}] decay=${FLOW_LAMBDA_EMA_DECAY} every_n=${FLOW_GRADNORM_EVERY_N} probe_samples=${FLOW_GRADNORM_PROBE_SAMPLES} jump_cap=${FLOW_LAMBDA_JUMP_CAP} warmup_steps=${FLOW_LAMBDA_WARMUP_STEPS} min_pixels=${FLOW_MIN_SUPERVISED_PIXELS}"
  echo "[launch] scene_flow: gt_dir='${FLOW_GT_DIR:-<dataset default>}' index='${FLOW_INDEX:-<dataset default>}'"
fi
[ "${#EXTRA_QWENVL_FLAGS[@]}" -gt 0 ] && echo "[launch] depth-image: stereo_extra_image_cam_id=${EXTRA_IMAGE_CAM_ID}"
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
  "${EXTRA_QWENVL_FLAGS[@]}" \
  "${SCENE_FLOW_FLAGS[@]}" \
  --framework.ffs_vlm_input.ffs_model_path ${ffs_model_path} \
  --framework.ffs_vlm_input.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_vlm_input.ffs_feature_source gru_hidden \
  --framework.ffs_vlm_input.gru_hidden_dim 16 \
  --framework.ffs_vlm_input.ffs_image_size 256 \
  --framework.ffs_vlm_input.inject_hidden_dim 256 \
  --framework.ffs_vlm_input.num_cameras 2 \
  --framework.ffs_vlm_input.left_ref_idx 1 \
  --framework.ffs_vlm_input.primary_view_idx 0 \
  --framework.ffs_vlm_input.inject_cam_id 1 \
  --framework.ffs_vlm_input.inject_gate_init ${INJECT_GATE_INIT} \
  --framework.ffs_controlnet.ffs_model_path ${ffs_model_path} \
  --framework.ffs_controlnet.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_controlnet.ffs_feature_source gru_hidden \
  --framework.ffs_controlnet.gru_hidden_dim 16 \
  --framework.ffs_controlnet.ffs_image_size 256 \
  --framework.ffs_controlnet.inject_hidden_dim 256 \
  --framework.ffs_controlnet.num_cameras 2 \
  --framework.ffs_controlnet.left_ref_idx 1 \
  --framework.ffs_controlnet.primary_view_idx 0 \
  --framework.ffs_controlnet.inject_cam_id 1 \
  --framework.ffs_controlnet.expected_vlm_layers 24 \
  --framework.ffs_vlm_controlnet.ffs_model_path ${ffs_model_path} \
  --framework.ffs_vlm_controlnet.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_vlm_controlnet.ffs_feature_source gru_hidden \
  --framework.ffs_vlm_controlnet.gru_hidden_dim 16 \
  --framework.ffs_vlm_controlnet.ffs_image_size 256 \
  --framework.ffs_vlm_controlnet.hint_hidden_dim 256 \
  --framework.ffs_vlm_controlnet.num_cameras 2 \
  --framework.ffs_vlm_controlnet.left_ref_idx 1 \
  --framework.ffs_vlm_controlnet.primary_view_idx 0 \
  --framework.ffs_vlm_controlnet.inject_cam_id 1 \
  --framework.ffs_depth_token.ffs_model_path ${ffs_model_path} \
  --framework.ffs_depth_token.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_depth_token.ffs_feature_source gru_hidden \
  --framework.ffs_depth_token.gru_hidden_dim 16 \
  --framework.ffs_depth_token.ffs_image_size 256 \
  --framework.ffs_depth_token.num_cameras 2 \
  --framework.ffs_depth_token.left_ref_idx 1 \
  --framework.ffs_depth_token.primary_view_idx 0 \
  --framework.ffs_depth_token.inject_cam_id 1 \
  --framework.ffs_depth_token.num_depth_tokens ${NUM_DEPTH_TOKENS} \
  --framework.ffs_depth_token.pool_hw ${POOL_HW} \
  --framework.ffs_depth_token.strip_depth_tokens ${STRIP_DEPTH_BOOL} \
  "${FFS_CACHE_FLAGS[@]}" \
  --framework.ffs_depth_image.ffs_model_path ${ffs_model_path} \
  --framework.ffs_depth_image.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_depth_image.ffs_feature_source gru_hidden \
  --framework.ffs_depth_image.gru_hidden_dim 16 \
  --framework.ffs_depth_image.ffs_image_size 256 \
  --framework.ffs_depth_image.num_cameras 2 \
  --framework.ffs_depth_image.left_ref_idx 1 \
  --framework.ffs_depth_image.primary_view_idx 0 \
  --framework.ffs_depth_image.inject_cam_id 1 \
  --framework.ffs_depth_image.depth_prompt "Below is the stereo disparity (depth) map of the left view:" \
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
  --wandb_project starVLA_Qwen0p8_GR00T_FFS \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
