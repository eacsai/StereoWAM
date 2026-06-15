#!/bin/bash
# Qwen3.5-0.8B + GR00T + Method #10B Utonia neutral-position resampler tokens.
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
DATA_MIX=${DATA_MIX:-libero_all_sfstereo_rightprimary}
FRAMEWORK=${FRAMEWORK:-QwenGR00T_UtoniaResamplerFFS}
PRETRAINED_CKPT=${PRETRAINED_CKPT-}

ffs_model_path=${FFS_MODEL_PATH:-${FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth}
ffs_sha256=${FFS_SHA256:-98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692}
utonia_ckpt_path=${UTONIA_CKPT_PATH:-./playground/Pretrained_models/Utonia/utonia.pth}

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-qwen3p5_0p8b_utonia_resampler_fromscratch_30k}

BS=${BS:-32}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
GPUS=${GPUS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
PORT=${PORT:-29763}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}
FREEZE_MODULES=${FREEZE_MODULES-}
TRAIN_ONLY=${TRAIN_ONLY-}
CAM_ROPE=${CAM_ROPE:-0}
CAM_BRANCH=${CAM_BRANCH:-0}
BACKPROJECT_STRIDE=${BACKPROJECT_STRIDE:-4}
UTONIA_SCALE=${UTONIA_SCALE:-4.0}
UTONIA_ENABLE_FLASH=${UTONIA_ENABLE_FLASH:-false}
UTONIA_R5_MAX_SEC_PER_STEP=${UTONIA_R5_MAX_SEC_PER_STEP:-10.0}
UTONIA_R5_FALLBACK_STRIDES=${UTONIA_R5_FALLBACK_STRIDES:-8,16}
RESAMPLER_LAYERS=${RESAMPLER_LAYERS:-1}
RESAMPLER_HEADS=${RESAMPLER_HEADS:-8}

case "${utonia_ckpt_path}" in
  /*) echo "[guard] UTONIA_CKPT_PATH must be relative, got '${utonia_ckpt_path}'"; exit 3 ;;
esac
case "${CAM_ROPE}" in 0|1) ;; *) echo "[guard] CAM_ROPE must be 0 or 1, got '${CAM_ROPE}'"; exit 3 ;; esac
case "${CAM_BRANCH}" in 0|1) ;; *) echo "[guard] CAM_BRANCH must be 0 or 1, got '${CAM_BRANCH}'"; exit 3 ;; esac
CAM_ROPE_BOOL=$([ "${CAM_ROPE}" = "1" ] && echo true || echo false)
CAM_BRANCH_BOOL=$([ "${CAM_BRANCH}" = "1" ] && echo true || echo false)

[ "${FRAMEWORK}" = "QwenGR00T_UtoniaResamplerFFS" ] || {
  echo "[guard] this launcher requires FRAMEWORK=QwenGR00T_UtoniaResamplerFFS, got '${FRAMEWORK}'"; exit 3;
}
[ "${run_id}" = "qwen3p5_0p8b_utonia_resampler_fromscratch_30k" ] || {
  echo "[guard] run_id must be qwen3p5_0p8b_utonia_resampler_fromscratch_30k, got '${run_id}'"; exit 3;
}
[ "${CAM_ROPE}" = "0" ] || { echo "[guard] Method #10 fromscratch runs use CAM_ROPE=0, got '${CAM_ROPE}'"; exit 3; }
[ "${CAM_BRANCH}" = "0" ] || { echo "[guard] Method #10 is incompatible with CAM_BRANCH=1"; exit 3; }
[ -z "${PRETRAINED_CKPT}" ] || { echo "[guard] fromscratch requires PRETRAINED_CKPT='', got '${PRETRAINED_CKPT}'"; exit 3; }
[ -z "${FREEZE_MODULES}" ] || { echo "[guard] fromscratch requires FREEZE_MODULES='', got '${FREEZE_MODULES}'"; exit 3; }
[ -z "${TRAIN_ONLY}" ] || { echo "[guard] fromscratch trains all params; TRAIN_ONLY must be empty, got '${TRAIN_ONLY}'"; exit 3; }

preflight_paths=("${ffs_model_path}" "${utonia_ckpt_path}" "${DS_CONFIG}")
for path in "${preflight_paths[@]}"; do
  [ -f "${path}" ] || { echo "[preflight] missing file: ${path}"; exit 1; }
done

${CONDA_VENV}/python - <<'PY'
import utonia, spconv, torch_scatter, addict
print("UTONIA_IMPORT_GATE_OK")
PY

run_required_smoke() {
  local smoke="$1"
  echo "[preflight] running mandatory smoke ${smoke}"
  set +e
  smoke_out=$(BASE_VLM="${base_vlm}" REQUIRE_BASE_VLM_SMOKE=1 ${CONDA_VENV}/python "${smoke}" 2>&1)
  smoke_status=$?
  set -e
  printf '%s\n' "$smoke_out"
  [ "$smoke_status" -eq 0 ] || { echo "[preflight] smoke failed: ${smoke}"; exit "$smoke_status"; }
  printf '%s\n' "$smoke_out" | grep -q 'SMOKE_ALL_PASS' || { echo "[preflight] smoke missing SMOKE_ALL_PASS: ${smoke}"; exit 1; }
}

run_required_smoke scripts/h100b/smoke_utonia_perpatch.py
run_required_smoke scripts/h100b/smoke_utonia_resampler.py

echo "[preflight] running mandatory Utonia R5 cost benchmark"
set +e
bench_out=$(CUDA_VISIBLE_DEVICES=${GPUS%%,*} ${CONDA_VENV}/python -m starVLA.model.modules.stereo.utonia_pointcloud benchmark \
  --ffs-model-path "${ffs_model_path}" \
  --utonia-ckpt-path "${utonia_ckpt_path}" \
  --ffs-repo-dir "${FFS_REPO_DIR}" \
  --ffs-expected-sha256 "${ffs_sha256}" \
  --utonia-scale "${UTONIA_SCALE}" \
  --image-size 256 \
  --backproject-stride "${BACKPROJECT_STRIDE}" \
  --fallback-strides "${UTONIA_R5_FALLBACK_STRIDES}" \
  --batch-sizes "1,4,8,${BS}" \
  --max-sec-per-step "${UTONIA_R5_MAX_SEC_PER_STEP}" 2>&1)
bench_status=$?
set -e
printf '%s\n' "$bench_out"
[ "$bench_status" -eq 0 ] || { echo "[preflight] Utonia R5 benchmark failed"; exit "$bench_status"; }
selected_stride=$(printf '%s\n' "$bench_out" | awk -F= '/UTONIA_COST_BENCH_SELECTED_STRIDE=/{print $2}' | tail -1)
BACKPROJECT_STRIDE=${selected_stride:-${BACKPROJECT_STRIDE}}

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

echo "[launch] ${FRAMEWORK} + Qwen3.5-0.8B + GR00T Utonia resampler tokens"
echo "[launch] DATA_ROOT=${DATA_ROOT} DATA_MIX=${DATA_MIX}"
echo "[launch] FREEZE_MODULES='${FREEZE_MODULES}' TRAIN_ONLY='${TRAIN_ONLY}'"
echo "[launch] cam_rope=${CAM_ROPE_BOOL} cam_branch=${CAM_BRANCH_BOOL} backproject_stride=${BACKPROJECT_STRIDE}"
DS_GA=$(printf '%s' "${DS_CONFIG}" | grep -oE 'ga[0-9]+' | grep -oE '[0-9]+' | tail -1); DS_GA=${DS_GA:-4}
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA${DS_GA} (${DS_CONFIG##*/}) = eff_$((BS*NUM_PROCESSES*DS_GA)) | MAX_STEPS=$MAX_STEPS | run_id=$run_id"

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
  --framework.utonia_pointcloud.primary_idx 1 \
  --framework.utonia_pointcloud.right_view_idx 0 \
  --framework.utonia_pointcloud.primary_cam_id 1 \
  --framework.utonia_pointcloud.num_point_tokens 64 \
  --framework.utonia_pointcloud.resampler_layers ${RESAMPLER_LAYERS} \
  --framework.utonia_pointcloud.resampler_heads ${RESAMPLER_HEADS} \
  --framework.utonia_pointcloud.gate_init zero \
  --datasets.vla_data.data_root_dir ${DATA_ROOT} \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.train_only "${TRAIN_ONLY}" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval $SAVE_INTERVAL \
  --trainer.logging_frequency 20 \
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
