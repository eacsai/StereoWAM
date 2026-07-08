#!/bin/bash
# Qwen3.5-0.8B + GR00T + Utonia prompt-token + scene_flow aux head (= ③ Utonia + ④ scene_flow).
# Based on run_qwen0p8_groot_utonia_prompttoken.sh + scene_flow args (online_grad_norm=false, flow_lambda=130 per spec §9).
# full ft (FREEZE_MODULES='') aligns ③; primary contrast = ③ (single-var = scene_flow on Utonia).
#
# Version B for the Version A comparison: same cached Utonia grid/source as
# run_qwen0p8_groot_utonia_perpatch.sh, but inserts 64 prompt rows before the
# left image token run instead of adding per-layer primary-token residuals.
set -e

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
FRAMEWORK=${FRAMEWORK:-QwenGR00T_UtoniaPromptTokenFFS}
PRETRAINED_CKPT=${PRETRAINED_CKPT-}

ffs_model_path=${FFS_MODEL_PATH:-${FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth}
ffs_sha256=${FFS_SHA256:-98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692}
utonia_ckpt_path=${UTONIA_CKPT_PATH:-./playground/Pretrained_models/Utonia/utonia.pth}
# UTONIA_LIVE=1 -> compute Utonia/FFS grids on the fly (no per-patch cache).
# Default 0 keeps the original cached A-vs-B comparison path unchanged.
UTONIA_LIVE=${UTONIA_LIVE:-0}
case "${UTONIA_LIVE}" in 0|1) ;; *) echo "[guard] UTONIA_LIVE must be 0 or 1, got "; exit 3 ;; esac
if [ "${UTONIA_LIVE}" = "1" ]; then
  UTONIA_CACHE_DIR=""   # empty -> model resolves utonia_cache_dir to None -> live compute path
else
  UTONIA_CACHE_DIR=${UTONIA_CACHE_DIR:-playground/Datasets/utonia_cache_perpatch}
fi

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-qwen3p5_0p8b_utonia_prompttoken_sceneflow_lambda130_leftprimary_fromscratch_fullft_eff128_maskfix_30k}

BS=${BS:-16}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
GPUS=${GPUS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT=${PORT:-29764}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga8.yaml}
FREEZE_MODULES=${FREEZE_MODULES-}
TRAIN_ONLY=${TRAIN_ONLY-}
CAM_ROPE=${CAM_ROPE:-0}
CAM_BRANCH=${CAM_BRANCH:-0}
CAM_BRANCH_HEADS=${CAM_BRANCH_HEADS:-4}
CAM_BRANCH_HEAD_DIM=${CAM_BRANCH_HEAD_DIM:-128}
BACKPROJECT_STRIDE=${BACKPROJECT_STRIDE:-4}
UTONIA_SCALE=${UTONIA_SCALE:-4.0}
UTONIA_ENABLE_FLASH=${UTONIA_ENABLE_FLASH:-false}
LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-20}

case "${utonia_ckpt_path}" in
  /*) echo "[guard] UTONIA_CKPT_PATH must be relative, got '${utonia_ckpt_path}'"; exit 3 ;;
esac
case "${CAM_ROPE}" in 0|1) ;; *) echo "[guard] CAM_ROPE must be 0 or 1, got '${CAM_ROPE}'"; exit 3 ;; esac
case "${CAM_BRANCH}" in 0|1) ;; *) echo "[guard] CAM_BRANCH must be 0 or 1, got '${CAM_BRANCH}'"; exit 3 ;; esac
CAM_ROPE_BOOL=$([ "${CAM_ROPE}" = "1" ] && echo true || echo false)
CAM_BRANCH_BOOL=$([ "${CAM_BRANCH}" = "1" ] && echo true || echo false)

[ "${FRAMEWORK}" = "QwenGR00T_UtoniaPromptTokenFFS" ] || {
  echo "[guard] this launcher requires FRAMEWORK=QwenGR00T_UtoniaPromptTokenFFS, got '${FRAMEWORK}'"; exit 3;
}
[ "${DATA_MIX}" = "libero_all_sfstereo_leftprimary" ] || {
  echo "[guard] cached Utonia run requires DATA_MIX=libero_all_sfstereo_leftprimary, got '${DATA_MIX}'"; exit 3;
}
[ "${CAM_ROPE}" = "0" ] || { echo "[guard] fair A-vs-B Method #10 run uses CAM_ROPE=0, got '${CAM_ROPE}'"; exit 3; }
case "${run_id}" in
  *cambranch*) [ "${CAM_BRANCH}" = "1" ] || { echo "[guard] run_id says cambranch but CAM_BRANCH!=1"; exit 3; } ;;
  *) [ "${CAM_BRANCH}" = "0" ] || { echo "[guard] CAM_BRANCH=1 requires run_id containing cambranch"; exit 3; } ;;
esac
[ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] from-scratch requires PRETRAINED_CKPT='', got '${PRETRAINED_CKPT}'"; exit 3; }
[ -z "${FREEZE_MODULES}" ] || { echo "[guard] full finetune requires FREEZE_MODULES='', got '${FREEZE_MODULES}'"; exit 3; }
[ -z "${TRAIN_ONLY}" ] || { echo "[guard] full finetune trains all params; TRAIN_ONLY must be empty, got '${TRAIN_ONLY}'"; exit 3; }
[ "${UTONIA_LIVE}" = "1" ] || [ -n "${UTONIA_CACHE_DIR}" ] || { echo "[guard] prompt-token comparison must use cached Utonia; set UTONIA_CACHE_DIR (or UTONIA_LIVE=1 for live FFS)"; exit 3; }
[ "${MAX_STEPS}" = "30000" ] || { echo "[guard] fair comparison uses MAX_STEPS=30000, got '${MAX_STEPS}'"; exit 3; }
# RELAXED 2026-06-25 (save freq不影响模型/30k对比, 低=抗A800抢占): [ "${SAVE_INTERVAL}" = "10000" ] || { echo "[guard] fair comparison uses SAVE_INTERVAL=10000, got '${SAVE_INTERVAL}'"; exit 3; }
[ "${WARMUP_STEPS}" = "5000" ] || { echo "[guard] fair comparison uses WARMUP_STEPS=5000, got '${WARMUP_STEPS}'"; exit 3; }

preflight_paths=("${ffs_model_path}" "${utonia_ckpt_path}" "${DS_CONFIG}" "${config_yaml}")
for path in "${preflight_paths[@]}"; do
  [ -f "${path}" ] || { echo "[preflight] missing file: ${path}"; exit 1; }
done
[ "${UTONIA_LIVE}" = "1" ] || [ -d "${UTONIA_CACHE_DIR}" ] || { echo "[preflight] missing UTONIA_CACHE_DIR: ${UTONIA_CACHE_DIR}"; exit 1; }

${CONDA_VENV}/python - <<'PY'
import utonia, spconv, torch_scatter, addict
print("UTONIA_IMPORT_GATE_OK")
PY

SMOKE_CAM_BRANCH_ARGS=()
if [ "${CAM_BRANCH}" = "1" ]; then
  SMOKE_CAM_BRANCH_ARGS+=(--cam-branch --cam-branch-heads "${CAM_BRANCH_HEADS}" --cam-branch-head-dim "${CAM_BRANCH_HEAD_DIM}")
fi

if [ "${SKIP_UTONIA_PROMPTTOKEN_SMOKE:-0}" != "1" ]; then
  echo "[preflight] running mandatory prompt-token smoke"
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} ${CONDA_VENV}/python scripts/a800/smoke_utonia_prompttoken.py \
    --mode fromscratch \
    --base-vlm "${base_vlm}" \
    --ffs-model-path "${ffs_model_path}" \
    --ffs-repo-dir "${FFS_REPO_DIR}" \
    --ffs-expected-sha256 "${ffs_sha256}" \
    --utonia-ckpt-path "${utonia_ckpt_path}" \
    --batch-size "${BS}" \
    --train-steps 2 \
    --device cuda \
    "${SMOKE_CAM_BRANCH_ARGS[@]}"
fi

if [ "${UTONIA_LIVE}" = "1" ]; then
  echo "[preflight] UTONIA_LIVE=1: skipping cache validation, computing Utonia/FFS grids on the fly"
else
UTONIA_CACHE_DIR="${UTONIA_CACHE_DIR}" \
DATA_ROOT="${DATA_ROOT}" \
DATA_MIX="${DATA_MIX}" \
CONFIG_YAML="${config_yaml}" \
FFS_MODEL_PATH="${ffs_model_path}" \
UTONIA_CKPT_PATH="${utonia_ckpt_path}" \
FFS_SHA256="${ffs_sha256}" \
UTONIA_SCALE="${UTONIA_SCALE}" \
UTONIA_ENABLE_FLASH="${UTONIA_ENABLE_FLASH}" \
BACKPROJECT_STRIDE="${BACKPROJECT_STRIDE}" \
${CONDA_VENV}/python - <<'PY'
import os
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.framework.VLM4A.QwenGR00T_UtoniaPromptTokenFFS import (
    QwenGR00TUtoniaPromptTokenFFSDefaultConfig,
)
from starVLA.model.modules.stereo.utonia_pointcloud import _sha256_file, validate_utonia_cache

cache_dir = Path(os.environ["UTONIA_CACHE_DIR"])
data_mix = os.environ["DATA_MIX"]
cfg = OmegaConf.load(os.environ["CONFIG_YAML"])
cfg = OmegaConf.merge(
    cfg,
    OmegaConf.create(
        {
            "datasets": {
                "vla_data": {
                    "data_root_dir": os.environ["DATA_ROOT"],
                    "data_mix": data_mix,
                }
            }
        }
    ),
)
cfg = apply_config_compat(cfg)
mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
suites = [name for name, _weight, _robot_type in mixture_spec]
dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train", seed=int(cfg.get("seed", 42)))
by_name = {single.dataset_name: single for single in dataset.datasets}

pc_cfg = dict(QwenGR00TUtoniaPromptTokenFFSDefaultConfig().utonia_pointcloud)
pc_cfg.update(
    {
        "ffs_model_path": os.environ["FFS_MODEL_PATH"],
        "ffs_expected_sha256": os.environ.get("FFS_SHA256") or None,
        "utonia_ckpt_path": os.environ["UTONIA_CKPT_PATH"],
        "utonia_cache_dir": str(cache_dir),
        "utonia_scale": float(os.environ["UTONIA_SCALE"]),
        "utonia_enable_flash": os.environ["UTONIA_ENABLE_FLASH"].lower() == "true",
        "ffs_image_size": 256,
        "image_width": 256,
        "image_height": 256,
        "backproject_stride": int(os.environ["BACKPROJECT_STRIDE"]),
        "fovy_degrees": 45.0,
        "baseline_m": 0.06,
        "depth_min": 0.05,
        "depth_max": 3.0,
        "disp_eps": 0.001,
        "num_cameras": 2,
        "left_ref_idx": 1,
        "primary_view_idx": 0,
        "inject_cam_id": 1,
        "num_point_tokens": 64,
        "gate_init": "zero",
    }
)
validate_utonia_cache(
    cache_dir,
    suites,
    dataset_or_none=by_name,
    ffs_sha256=_sha256_file(os.environ["FFS_MODEL_PATH"]),
    utonia_sha256=_sha256_file(os.environ["UTONIA_CKPT_PATH"]),
    pc_cfg=pc_cfg,
    grid_hw=(8, 8),
    require_all_done=True,
    expected_data_root=str(cfg.datasets.vla_data.data_root_dir),
    expected_data_mix=data_mix,
)
print("UTONIA_CACHE_PREFLIGHT_OK")
PY
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}/checkpoints"
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

METHOD10_CKPT_MAX_BYTES=${METHOD10_CKPT_MAX_BYTES:-2600000000}
METHOD10_CKPT_AUDIT_POLL_SEC=${METHOD10_CKPT_AUDIT_POLL_SEC:-60}
METHOD10_CKPT_AUDIT_STABILITY_SEC=${METHOD10_CKPT_AUDIT_STABILITY_SEC:-10}

method10_ckpt_audit() {
  local ckpt_dir="$1"
  local ok_file="$2"
  local fail_file="$3"
  METHOD10_CKPT_MAX_BYTES="${METHOD10_CKPT_MAX_BYTES}" \
  METHOD10_CKPT_AUDIT_POLL_SEC="${METHOD10_CKPT_AUDIT_POLL_SEC}" \
  METHOD10_CKPT_AUDIT_STABILITY_SEC="${METHOD10_CKPT_AUDIT_STABILITY_SEC}" \
  ${CONDA_VENV}/python - "$ckpt_dir" "$ok_file" "$fail_file" <<'PY'
import glob
import os
import sys
import time

import torch

ckpt_dir, ok_file, fail_file = sys.argv[1:4]
max_bytes = int(os.environ.get("METHOD10_CKPT_MAX_BYTES", "2600000000"))
poll_sec = int(os.environ.get("METHOD10_CKPT_AUDIT_POLL_SEC", "60"))
stability_sec = int(os.environ.get("METHOD10_CKPT_AUDIT_STABILITY_SEC", "10"))


def fail(msg, code=31):
    with open(fail_file, "w") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)
    raise SystemExit(code)


while True:
    files = sorted(
        glob.glob(os.path.join(ckpt_dir, "steps_*_pytorch_model.pt")),
        key=os.path.getmtime,
    )
    if files:
        path = files[0]
        last_size = -1
        stable_reads = 0
        while stable_reads < 2:
            size = os.path.getsize(path)
            if size > 0 and size == last_size:
                stable_reads += 1
            else:
                stable_reads = 0
                last_size = size
            if stable_reads < 2:
                time.sleep(stability_sec)
        size = last_size
        last_exc = None
        state = None
        for _attempt in range(12):
            try:
                state = torch.load(path, map_location="cpu")
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(10)
        if state is None:
            fail(f"METHOD10_CKPT_AUDIT_FAIL load_error path={path} error={last_exc}")
        if not isinstance(state, dict):
            fail(f"METHOD10_CKPT_AUDIT_FAIL state_dict_type={type(state).__name__} path={path}")
        bad = [
            key for key in state.keys()
            if key.startswith(("utonia.", "ffs.")) or ".utonia." in key or ".ffs." in key
        ]
        if bad:
            fail(f"METHOD10_CKPT_AUDIT_FAIL frozen encoder keys leaked: {bad[:8]}")
        if size > max_bytes:
            fail(f"METHOD10_CKPT_AUDIT_FAIL size_bytes={size} budget={max_bytes} path={path}")
        msg = f"METHOD10_CKPT_AUDIT_OK path={path} size_bytes={size} budget={max_bytes}"
        with open(ok_file, "w") as fh:
            fh.write(msg + "\n")
        print(msg, flush=True)
        raise SystemExit(0)
    time.sleep(poll_sec)
PY
}

echo "[launch] ${FRAMEWORK} + Qwen3.5-0.8B + GR00T Utonia prompt tokens"
echo "[launch] cached Utonia grid/source='${UTONIA_CACHE_DIR}', injection topology=prompt-token"
echo "[launch] DATA_ROOT=${DATA_ROOT} DATA_MIX=${DATA_MIX}"
echo "[launch] FREEZE_MODULES='${FREEZE_MODULES}' TRAIN_ONLY='${TRAIN_ONLY}' PRETRAINED_CKPT='${PRETRAINED_CKPT}'"
echo "[launch] cam_rope=${CAM_ROPE_BOOL} cam_branch=${CAM_BRANCH_BOOL} backproject_stride=${BACKPROJECT_STRIDE}"
DS_GA=$(printf '%s' "${DS_CONFIG}" | grep -oE 'ga[0-9]+' | grep -oE '[0-9]+' | tail -1); DS_GA=${DS_GA:-4}
EFFECTIVE_BATCH=$((BS*NUM_PROCESSES*DS_GA))
[ "${EFFECTIVE_BATCH}" = "128" ] || { echo "[guard] fair comparison requires effective batch 128, got ${EFFECTIVE_BATCH} (BS=$BS NUM_PROCESSES=$NUM_PROCESSES GA=$DS_GA)"; exit 3; }
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA${DS_GA} (${DS_CONFIG##*/}) = eff_${EFFECTIVE_BATCH} | MAX_STEPS=$MAX_STEPS | SAVE_INTERVAL=$SAVE_INTERVAL | WARMUP=$WARMUP_STEPS | run_id=$run_id"

audit_ok="${output_dir}/checkpoints/.method10_first_ckpt_audit.ok"
audit_fail="${output_dir}/checkpoints/.method10_first_ckpt_audit.fail"
rm -f "$audit_ok" "$audit_fail"
method10_ckpt_audit "${output_dir}/checkpoints" "$audit_ok" "$audit_fail" &
audit_pid=$!

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
  --framework.action_model.diffusion_model_cfg.interleave_self_attention true \
  --framework.utonia_pointcloud.ffs_model_path ${ffs_model_path} \
  --framework.utonia_pointcloud.ffs_expected_sha256 ${ffs_sha256} \
  --framework.utonia_pointcloud.ffs_feature_source gru_hidden \
  --framework.utonia_pointcloud.gru_hidden_dim 16 \
  --framework.utonia_pointcloud.ffs_image_size 256 \
  --framework.utonia_pointcloud.utonia_ckpt_path ${utonia_ckpt_path} \
  --framework.utonia_pointcloud.utonia_scale ${UTONIA_SCALE} \
  --framework.utonia_pointcloud.utonia_enable_flash ${UTONIA_ENABLE_FLASH} \
  --framework.utonia_pointcloud.fovy_degrees 45.0 \
  --framework.utonia_pointcloud.baseline_m 0.06 \
  --framework.utonia_pointcloud.image_width 256 \
  --framework.utonia_pointcloud.image_height 256 \
  --framework.utonia_pointcloud.backproject_stride ${BACKPROJECT_STRIDE} \
  --framework.utonia_pointcloud.depth_min 0.05 \
  --framework.utonia_pointcloud.depth_max 3.0 \
  --framework.utonia_pointcloud.disp_eps 0.001 \
  --framework.utonia_pointcloud.num_cameras 2 \
  --framework.utonia_pointcloud.left_ref_idx 1 \
  --framework.utonia_pointcloud.primary_view_idx 0 \
  --framework.utonia_pointcloud.inject_cam_id 1 \
  --framework.utonia_pointcloud.num_point_tokens 64 \
  --framework.utonia_pointcloud.point_prompt "Left image point-cloud features:" \
  --framework.utonia_pointcloud.gate_init zero \
  --framework.utonia_pointcloud.utonia_cache_dir "${UTONIA_CACHE_DIR}" \
  --framework.action_model.scene_flow.enabled true \
  --framework.action_model.scene_flow.flow_lambda 130 \
  --framework.action_model.scene_flow.online_grad_norm_enabled false \
  --framework.action_model.scene_flow.target_grad_ratio 0.05 \
  --framework.action_model.scene_flow.lambda_min 0.001 \
  --framework.action_model.scene_flow.lambda_max 10000 \
  --framework.action_model.scene_flow.lambda_ema_decay 0.97 \
  --framework.action_model.scene_flow.online_grad_norm_every_n_steps 10 \
  --framework.action_model.scene_flow.probe_samples 4 \
  --framework.action_model.scene_flow.lambda_jump_cap 2.0 \
  --framework.action_model.scene_flow.lambda_warmup_steps 100 \
  --framework.action_model.scene_flow.min_supervised_pixels 128 \
  --framework.action_model.scene_flow.grid_size 16 \
  --framework.action_model.scene_flow.hidden_layer -1 \
  --framework.action_model.scene_flow.mask_mode dynamic \
  --framework.action_model.scene_flow.step0_flow_warmup_steps 1 \
  --framework.action_model.scene_flow.step0_action_loss_audit true \
  --framework.action_model.scene_flow.grad_ratio_steps 20 \
  --datasets.vla_data.scene_flow.enabled true \
  --datasets.vla_data.scene_flow.gt_only_sampler true \
  --datasets.vla_data.scene_flow.expected_sidecar_to_training_flip rot180 \
  --datasets.vla_data.data_root_dir ${DATA_ROOT} \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.train_only "${TRAIN_ONLY}" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval $SAVE_INTERVAL \
  --trainer.num_warmup_steps $WARMUP_STEPS \
  --trainer.logging_frequency ${LOGGING_FREQUENCY} \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Qwen0p8_GR00T_Utonia \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}" &
train_pid=$!

audit_done=0
while kill -0 "$train_pid" 2>/dev/null; do
  if [ -f "$audit_fail" ]; then
    cat "$audit_fail"
    kill "$train_pid" 2>/dev/null || true
    wait "$train_pid" 2>/dev/null || true
    kill "$audit_pid" 2>/dev/null || true
    wait "$audit_pid" 2>/dev/null || true
    exit 31
  fi
  if [ -f "$audit_ok" ]; then
    wait "$audit_pid"
    audit_done=1
    break
  fi
  sleep 30
done

set +e
wait "$train_pid"
train_status=$?
set -e
if [ -f "$audit_fail" ]; then
  cat "$audit_fail"
  exit 31
fi
if [ "$audit_done" = "0" ] && kill -0 "$audit_pid" 2>/dev/null; then
  kill "$audit_pid" 2>/dev/null || true
  wait "$audit_pid" 2>/dev/null || true
fi
exit "$train_status"
