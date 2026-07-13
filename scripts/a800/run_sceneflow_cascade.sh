#!/bin/bash
# Scene-flow dual-DiT cascade (spec: sceneflow-dualdit-cascade). Built on 4090d (authoritative).
# NO Utonia / NO FFS / NO orthogrid: the shared VLM eats RAW leftprimary stereo (2 views:
# right_view + primary). Its last hidden feeds BOTH a scene-flow DiT and the GR00T action DiT;
# the action DiT reads the scene DiT's tapped mid-late hidden through a ZERO-INIT gated cross-attn.
# Runs on a800 or 4090d (set REPO_DIR).
#
# STAGE (required) selects the recipe (descriptive, not numbered):
#   joint_cascade       : train EVERYTHING jointly (VLM + action DiT + scene DiT + coupler),
#                         loss_mode=joint (action + FLOW_LAMBDA*flow). The zero-init coupler makes
#                         step-0 == plain baseline, then ramps. Supports FROM-SCRATCH (PRETRAINED_CKPT
#                         empty) — directly comparable to the from-scratch leftprimary table.
#   flow_predictor_only : TRAIN_ONLY=scene_predictor (FREEZE VLM + action head + coupler),
#                         loss_mode=flow_only. Trains ONLY the scene DiT. Because it FREEZES the VLM,
#                         it MUST warm-start a pretrained VLM (PRETRAINED_CKPT required — freezing a
#                         random VLM = learning flow from garbage). Use joint_cascade for from-scratch.
#
# TARGET_KEY selects the arm (single-variable A/B — same architecture, only the target differs):
#   flow_gt     (default)  predicted future scene flow (dataloader already emits it)
#   pointmap_gt            current-frame camera-frame XYZ pointmap = static-geometry twin
#                          (needs the current-pointmap GT sidecars; see GAP #2 GT-gen)
set -e

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "${REPO_DIR:-}" ]; then
  if [ -f "./starVLA/training/train_starvla.py" ]; then
    REPO_DIR=$(pwd)                                      # safe when launched from the authoritative repo
  else
    REPO_DIR=/home/wangqiwei/ICLR2026/starVLA            # a800 default fallback
  fi
fi
cd "${REPO_DIR}"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

CONDA_VENV=${CONDA_VENV:-$(pwd)/.venv/bin}
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-0.8B}

DATA_ROOT=${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW}
DATA_MIX=${DATA_MIX:-libero_all_sfstereo_leftprimary}
FRAMEWORK=QwenGR00T

# ---- cascade knobs ----
STAGE=${STAGE:?STAGE required: joint_cascade | flow_predictor_only}
TARGET_KEY=${TARGET_KEY:-flow_gt}
TAP_HIDDEN_INDEX=${TAP_HIDDEN_INDEX:-10}
FLOW_LAMBDA=${FLOW_LAMBDA:-0.1}
DETACH_CONDITIONING=${DETACH_CONDITIONING:-true}
GRID_SIZE=${GRID_SIZE:-16}
Z_WEIGHT=${Z_WEIGHT:-2.0}
SIDECAR_FLIP=${SIDECAR_FLIP:-rot180}                      # leftprimary un-rotate (matches sceneflow launcher)

# ---- stage dispatch: freeze policy + loss mode + warm-start requirement ----
case "${STAGE}" in
  joint_cascade)
    LOSS_MODE=joint
    TRAIN_ONLY=${TRAIN_ONLY-}                             # default: train everything
    FREEZE_MODULES=${FREEZE_MODULES-}
    REQUIRE_WARMSTART=0                                   # from-scratch OK (PRETRAINED_CKPT may be empty)
    ;;
  flow_predictor_only)
    LOSS_MODE=flow_only
    TRAIN_ONLY=${TRAIN_ONLY:-scene_predictor}             # single freeze policy: only scene DiT trains
    FREEZE_MODULES=""                                     # (train_only overrides freeze_modules anyway)
    REQUIRE_WARMSTART=1                                   # freezes the VLM -> MUST warm-start a pretrained VLM
    ;;
  learn_scene_flow)
    LOSS_MODE=flow_only
    TRAIN_ONLY=${TRAIN_ONLY:-qwen_vl_interface,scene_predictor,stereo_cam_branch_layers_modules}  # train VLM + cam_branch + scene DiT; freeze action head + coupler
    FREEZE_MODULES=""
    REQUIRE_WARMSTART=0                                          # from-scratch stage-1 (no plain ckpt to freeze)
    ;;
  *) echo "[guard] STAGE must be joint_cascade | flow_predictor_only | learn_scene_flow, got '${STAGE}'"; exit 3 ;;
esac

PRETRAINED_CKPT=${PRETRAINED_CKPT-}

# ---- geometry arm (same in both A/B arms) ----
CAM_BRANCH=${CAM_BRANCH:-0}
CAM_BRANCH_HEADS=${CAM_BRANCH_HEADS:-4}
CAM_BRANCH_HEAD_DIM=${CAM_BRANCH_HEAD_DIM:-128}
case "${CAM_BRANCH}" in 0|1) ;; *) echo "[guard] CAM_BRANCH must be 0 or 1, got '${CAM_BRANCH}'"; exit 3 ;; esac
CAM_BRANCH_BOOL=$([ "${CAM_BRANCH}" = "1" ] && echo true || echo false)

# ---- descriptive run_id (arm + geometry + stage + warm-start-vs-fromscratch) ----
run_root_dir=./playground/Checkpoints
_arm=$([ "${TARGET_KEY}" = "pointmap_gt" ] && echo staticgeom || echo motion)
_geo=$([ "${CAM_BRANCH}" = "1" ] && echo cambranch || echo plain)
_ws=$([ -z "${PRETRAINED_CKPT}" ] && echo fromscratch || echo warmstart)
run_id=${RUN_ID:-qwen0p8_groot_cascade_${_arm}_${_geo}_${STAGE}_${_ws}_leftprimary}

BS=${BS:-16}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
GPUS=${GPUS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT=${PORT:-29775}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga8.yaml}
LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-20}

_bool01() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) echo 1 ;;
    0|false|FALSE|no|NO|n|N|off|OFF|"") echo 0 ;;
    *) echo "[guard] $2 must be boolean-like, got '$1'" >&2; exit 3 ;;
  esac
}
PAST_FLOW=$(_bool01 "${PAST_FLOW:-0}" PAST_FLOW)
PAST_FLOW_FULL_INJECT=$(_bool01 "${PAST_FLOW_FULL_INJECT:-0}" PAST_FLOW_FULL_INJECT)
PAST_FLOW_K=${PAST_FLOW_K:-2}
PAST_FLOW_DELTA=${PAST_FLOW_DELTA:-1}
PAST_FLOW_DROPOUT=${PAST_FLOW_DROPOUT:-0.3}
PAST_FLOW_NOISE=${PAST_FLOW_NOISE:-0.0}

# ---- dynamic-token focus knobs (default no-op) ----
DYNAMIC_LOSS_WEIGHT=${DYNAMIC_LOSS_WEIGHT:-1.0}
STATIC_ZERO_LOSS_WEIGHT=${STATIC_ZERO_LOSS_WEIGHT:-0.0}
DYNAMIC_DIRECTION_LOSS_WEIGHT=${DYNAMIC_DIRECTION_LOSS_WEIGHT:-0.0}
DYNAMIC_MAGNITUDE_LOSS_WEIGHT=${DYNAMIC_MAGNITUDE_LOSS_WEIGHT:-0.0}
DIRECTION_LOSS_EPS=${DIRECTION_LOSS_EPS:-1e-6}
CONDITIONING_PAST_MOTION_GATE=$(_bool01 "${CONDITIONING_PAST_MOTION_GATE:-0}" CONDITIONING_PAST_MOTION_GATE)
CONDITIONING_STATIC_SCALE=${CONDITIONING_STATIC_SCALE:-1.0}
CONDITIONING_MOTION_THRESHOLD=${CONDITIONING_MOTION_THRESHOLD:-1e-5}

# ---- fail-closed guards ----
[ "${FRAMEWORK}" = "QwenGR00T" ] || { echo "[guard] cascade requires FRAMEWORK=QwenGR00T"; exit 3; }
case "${TARGET_KEY}" in flow_gt|pointmap_gt) ;; *) echo "[guard] TARGET_KEY must be flow_gt | pointmap_gt, got '${TARGET_KEY}'"; exit 3 ;; esac
case "${LOSS_MODE}" in flow_only|joint) ;; *) echo "[guard] LOSS_MODE derivation bug: '${LOSS_MODE}'"; exit 3 ;; esac
if [ "${CONDITIONING_PAST_MOTION_GATE}" = "1" ] && [ "${PAST_FLOW}" != "1" ]; then
  echo "[guard] CONDITIONING_PAST_MOTION_GATE=1 requires PAST_FLOW=1 (gate is based on historical past flow)"; exit 3
fi
"${CONDA_VENV}/python" - <<PY
import math
try:
    dyn = float("${DYNAMIC_LOSS_WEIGHT}")
    stat = float("${STATIC_ZERO_LOSS_WEIGHT}")
    direction = float("${DYNAMIC_DIRECTION_LOSS_WEIGHT}")
    magnitude = float("${DYNAMIC_MAGNITUDE_LOSS_WEIGHT}")
    direction_eps = float("${DIRECTION_LOSS_EPS}")
    scale = float("${CONDITIONING_STATIC_SCALE}")
    thresh = float("${CONDITIONING_MOTION_THRESHOLD}")
except ValueError as exc:
    raise SystemExit(f"[guard] dynamic-focus knob must be numeric: {exc}")
for name, value in (
    ("DYNAMIC_LOSS_WEIGHT", dyn),
    ("STATIC_ZERO_LOSS_WEIGHT", stat),
    ("DYNAMIC_DIRECTION_LOSS_WEIGHT", direction),
    ("DYNAMIC_MAGNITUDE_LOSS_WEIGHT", magnitude),
    ("DIRECTION_LOSS_EPS", direction_eps),
    ("CONDITIONING_STATIC_SCALE", scale),
    ("CONDITIONING_MOTION_THRESHOLD", thresh),
):
    if not math.isfinite(value):
        raise SystemExit(f"[guard] {name} must be finite")
if dyn < 0 or stat < 0 or direction < 0 or magnitude < 0 or direction_eps < 0:
    raise SystemExit("[guard] scene-flow loss weights and DIRECTION_LOSS_EPS must be non-negative")
if dyn == 0 and stat == 0 and direction == 0 and magnitude == 0:
    raise SystemExit("[guard] at least one scene-flow loss weight must be > 0")
if not (0.0 <= scale <= 1.0):
    raise SystemExit("[guard] CONDITIONING_STATIC_SCALE must be in [0, 1]")
if thresh < 0:
    raise SystemExit("[guard] CONDITIONING_MOTION_THRESHOLD must be non-negative")
if "${TARGET_KEY}" != "flow_gt" and (stat > 0 or direction > 0 or magnitude > 0 or "${CONDITIONING_PAST_MOTION_GATE}" == "1"):
    raise SystemExit("[guard] dynamic-token focus/static-zero/aux losses/past-motion gate are only supported for TARGET_KEY=flow_gt")
PY
if [ "${REQUIRE_WARMSTART}" = "1" ]; then
  [ -n "${PRETRAINED_CKPT}" ] || { echo "[guard] STAGE=flow_predictor_only FREEZES the VLM -> requires PRETRAINED_CKPT (a pretrained/plain stage-0 ckpt). From-scratch is meaningless here (frozen random VLM). Use STAGE=joint_cascade for from-scratch."; exit 3; }
fi
[ -z "${PRETRAINED_CKPT}" ] || [ -f "${PRETRAINED_CKPT}" ] || { echo "[preflight] PRETRAINED_CKPT set but missing: ${PRETRAINED_CKPT}"; exit 1; }
case "${run_id}" in *cambranch*) [ "${CAM_BRANCH}" = "1" ] || { echo "[guard] run_id says cambranch but CAM_BRANCH!=1"; exit 3; } ;; esac
if [ "${PAST_FLOW_FULL_INJECT}" = "1" ]; then
  [ "${PAST_FLOW}" = "1" ] || { echo "[guard] PAST_FLOW_FULL_INJECT=1 requires PAST_FLOW=1"; exit 3; }
  case "${STAGE}" in learn_scene_flow|joint_cascade) ;; *) echo "[guard] PAST_FLOW_FULL_INJECT=1 is allowed only for STAGE=learn_scene_flow or joint_cascade"; exit 3 ;; esac
  case "${run_id}" in *fullinject*) ;; *) echo "[guard] PAST_FLOW_FULL_INJECT=1 requires run_id to contain fullinject"; exit 3 ;; esac
fi
case "${run_id}" in *fullinject*)
  [ "${PAST_FLOW}" = "1" ] && [ "${PAST_FLOW_FULL_INJECT}" = "1" ] || { echo "[guard] run_id says fullinject but PAST_FLOW/PAST_FLOW_FULL_INJECT are not both enabled"; exit 3; }
  ;;
esac
# staticgeom twin needs the current-pointmap sidecars; refuse until they exist (GAP #2)
if [ "${TARGET_KEY}" = "pointmap_gt" ] && [ "${ALLOW_MISSING_POINTMAP_GT:-0}" != "1" ]; then
  echo "[guard] TARGET_KEY=pointmap_gt needs current-pointmap GT sidecars (not yet generated). Set ALLOW_MISSING_POINTMAP_GT=1 to override once they exist."; exit 3
fi

preflight_paths=("${DS_CONFIG}" "${config_yaml}")
[ -n "${PRETRAINED_CKPT}" ] && preflight_paths+=("${PRETRAINED_CKPT}")
for path in "${preflight_paths[@]}"; do [ -f "${path}" ] || { echo "[preflight] missing file: ${path}"; exit 1; }; done

# ---- mandatory step-0 identity gate (skippable for a re-launch you've already gated) ----
if [ "${SKIP_CASCADE_SMOKE:-0}" != "1" ]; then
  echo "[preflight] running mandatory full-VLM step-0 cascade smoke (STEP0_EQ_BASELINE gate)"
  SMOKE_CAM=(); [ "${CAM_BRANCH}" = "1" ] && SMOKE_CAM=(--cam-branch)
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} ${CONDA_VENV}/python scripts/4090d/smoke_scene_cascade_fullvlm.py \
    --base-vlm "${base_vlm}" --config-yaml "${config_yaml}" \
    --target-key "${TARGET_KEY}" --tap-hidden-index "${TAP_HIDDEN_INDEX}" \
    --batch-size 2 --skip-warmstart "${SMOKE_CAM[@]}" \
    || { echo "[preflight] cascade step-0 smoke FAILED — refusing to launch"; exit 33; }
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}/checkpoints"
cp "$0" "${output_dir}/" 2>/dev/null || true

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
[ "${#existing_ckpts[@]}" -gt 0 ] && [ "${RESUME:-0}" != "1" ] && { echo "refuse: ckpts exist in ${output_dir}, set RESUME=1"; exit 1; }
if pgrep -f "run_id ${run_id}\$" >/dev/null 2>&1 || pgrep -f "run_id ${run_id} " >/dev/null 2>&1; then
  echo "refuse: a live process is already training run_id ${run_id}"; exit 1; fi
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true)

# --- past-flow ControlNet (temporal disambiguation): PAST_FLOW=1 enables it; mirrors the
# config into BOTH the model (framework.scene_predictor) and dataloader (datasets.vla_data.
# scene_flow) namespaces. K=n_past_steps, DELTA=one-step-flow spacing, DROPOUT/NOISE = the
# train/infer-gap defenses (past-flow dropout + noise-aug). ---
PAST_FLOW_ARGS=()
if [ "${PAST_FLOW}" = "1" ]; then
  PAST_FLOW_ARGS=(
    --framework.scene_predictor.past_flow_controlnet.enabled true
    --framework.scene_predictor.past_flow_controlnet.n_past_steps ${PAST_FLOW_K}
    --framework.scene_predictor.past_flow_controlnet.spacing_delta ${PAST_FLOW_DELTA}
    --framework.scene_predictor.past_flow_controlnet.dropout_p ${PAST_FLOW_DROPOUT}
    --framework.scene_predictor.past_flow_controlnet.noise_aug_std ${PAST_FLOW_NOISE}
    --framework.scene_predictor.past_flow_controlnet.full_injection $([ "${PAST_FLOW_FULL_INJECT}" = 1 ] && echo true || echo false)
    --datasets.vla_data.scene_flow.past_flow_controlnet.enabled true
    --datasets.vla_data.scene_flow.past_flow_controlnet.n_past_steps ${PAST_FLOW_K}
    --datasets.vla_data.scene_flow.past_flow_controlnet.spacing_delta ${PAST_FLOW_DELTA}
  )
fi

# warm-start flag only when a ckpt is given (empty => from-scratch: pass NO pretrained_checkpoint)
CKPT_FLAG=()
[ -n "${PRETRAINED_CKPT}" ] && CKPT_FLAG=(--trainer.pretrained_checkpoint "${PRETRAINED_CKPT}")

DS_GA=$(printf '%s' "${DS_CONFIG}" | grep -oE 'ga[0-9]+' | grep -oE '[0-9]+' | tail -1); DS_GA=${DS_GA:-4}
EFFECTIVE_BATCH=$((BS*NUM_PROCESSES*DS_GA))
# eff-batch discipline only on real runs; MAX_STEPS<1000 = gpu-smoke dry-run bypasses it.
if [ "${MAX_STEPS}" -ge 1000 ] && [ "${EFFECTIVE_BATCH}" != "128" ]; then
  echo "[guard] real run wants effective batch 128, got ${EFFECTIVE_BATCH} (BS=$BS NUM_PROCESSES=$NUM_PROCESSES GA=$DS_GA). Set BATCH knobs or MAX_STEPS<1000 for a smoke."; exit 3
fi

echo "[launch] SCENE-FLOW CASCADE | STAGE=${STAGE} arm=${_arm}(${TARGET_KEY}) loss_mode=${LOSS_MODE} tap=${TAP_HIDDEN_INDEX} detach=${DETACH_CONDITIONING} lambda=${FLOW_LAMBDA} past_flow_full_inject=${PAST_FLOW_FULL_INJECT}"
echo "[launch] dynamic_focus dyn_w=${DYNAMIC_LOSS_WEIGHT} static_zero_w=${STATIC_ZERO_LOSS_WEIGHT} dir_w=${DYNAMIC_DIRECTION_LOSS_WEIGHT} mag_w=${DYNAMIC_MAGNITUDE_LOSS_WEIGHT} dir_eps=${DIRECTION_LOSS_EPS} past_motion_gate=${CONDITIONING_PAST_MOTION_GATE} static_scale=${CONDITIONING_STATIC_SCALE} motion_thresh=${CONDITIONING_MOTION_THRESHOLD}"
echo "[launch] warmstart='${PRETRAINED_CKPT:-<from-scratch>}' TRAIN_ONLY='${TRAIN_ONLY}' FREEZE_MODULES='${FREEZE_MODULES}' cam_branch=${CAM_BRANCH_BOOL}"
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA${DS_GA} = eff_${EFFECTIVE_BATCH} | MAX_STEPS=$MAX_STEPS save=$SAVE_INTERVAL warmup=$WARMUP_STEPS | run_id=$run_id"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${FRAMEWORK} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled false \
  --framework.qwenvl.stereo_cam_rope_num_cameras 2 \
  --framework.qwenvl.stereo_cam_rope_baseline_m 0.06 \
  --framework.qwenvl.stereo_cam_rope_fovy_degrees 45.0 \
  --framework.qwenvl.stereo_cam_rope_image_width 256 \
  --framework.qwenvl.stereo_cam_rope_image_height 256 \
  --framework.qwenvl.stereo_cam_rope_spatial_merge 2 \
  --framework.qwenvl.stereo_cam_rope_right_first true \
  --framework.qwenvl.stereo_epipolar_mask_enabled false \
  --framework.qwenvl.stereo_cam_branch_enabled ${CAM_BRANCH_BOOL} \
  --framework.qwenvl.stereo_cam_branch_heads ${CAM_BRANCH_HEADS} \
  --framework.qwenvl.stereo_cam_branch_head_dim ${CAM_BRANCH_HEAD_DIM} \
  --framework.action_model.diffusion_model_cfg.interleave_self_attention true \
  --framework.scene_predictor.enabled true \
  --framework.scene_predictor.target_key ${TARGET_KEY} \
  --framework.scene_predictor.tap_hidden_index ${TAP_HIDDEN_INDEX} \
  --framework.scene_predictor.grid_size ${GRID_SIZE} \
  --framework.scene_predictor.z_weight ${Z_WEIGHT} \
  --framework.scene_predictor.dynamic_loss_weight ${DYNAMIC_LOSS_WEIGHT} \
  --framework.scene_predictor.static_zero_loss_weight ${STATIC_ZERO_LOSS_WEIGHT} \
  --framework.scene_predictor.dynamic_direction_loss_weight ${DYNAMIC_DIRECTION_LOSS_WEIGHT} \
  --framework.scene_predictor.dynamic_magnitude_loss_weight ${DYNAMIC_MAGNITUDE_LOSS_WEIGHT} \
  --framework.scene_predictor.direction_loss_eps ${DIRECTION_LOSS_EPS} \
  --framework.scene_predictor.conditioning_past_motion_gate $([ "${CONDITIONING_PAST_MOTION_GATE}" = 1 ] && echo true || echo false) \
  --framework.scene_predictor.conditioning_static_scale ${CONDITIONING_STATIC_SCALE} \
  --framework.scene_predictor.conditioning_motion_threshold ${CONDITIONING_MOTION_THRESHOLD} \
  --framework.scene_predictor.detach_conditioning ${DETACH_CONDITIONING} \
  --framework.scene_predictor.flow_lambda ${FLOW_LAMBDA} \
  --datasets.vla_data.scene_flow.enabled true \
  --datasets.vla_data.scene_flow.gt_only_sampler true \
  --datasets.vla_data.scene_flow.expected_sidecar_to_training_flip ${SIDECAR_FLIP} \
  "${PAST_FLOW_ARGS[@]}" \
  --datasets.vla_data.data_root_dir ${DATA_ROOT} \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.loss_mode ${LOSS_MODE} \
  "${CKPT_FLAG[@]}" \
  --trainer.train_only "${TRAIN_ONLY}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.num_warmup_steps $WARMUP_STEPS \
  --trainer.save_interval $SAVE_INTERVAL \
  --trainer.logging_frequency ${LOGGING_FREQUENCY} \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Qwen0p8_GR00T_SceneFlowCascade \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
