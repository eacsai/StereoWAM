#!/usr/bin/env bash
set -euo pipefail

# E5-full-only: E3 Camera Branch on all six Qwen3.5 full-attention layers plus
# sparse same-row cross-view E5 on [11,15,19,23].  The 18 Linear-Attention
# GatedDeltaNet layers are untouched.  This launcher is intentionally
# from-scratch and refuses every resume/pretrained entry point.

cd /home/wangqw/ICLR27/starVLA
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDA_VENV=${CONDA_VENV:-/home/wangqw/ICLR27/starVLA/.venv/bin}
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${BASE_VLM:-/home/wangqw/ICLR27/starVLA/playground/Pretrained_models/Qwen3.5-0.8B}
DATA_ROOT=${DATA_ROOT:-/home/wangqw/ICLR27/starVLA/playground/Datasets/LEROBOT_LIBERO_STEREO_LEFTPRIMARY}
DATA_MIX=${DATA_MIX:-libero_all_sfstereo_leftprimary}
FRAMEWORK=QwenGR00T
BS=${BS:-8}
MAX_STEPS=${MAX_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
NUM_PROCESSES=${NUM_PROCESSES:-4}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}
run_root_dir=${RUN_ROOT_DIR:-./playground/Checkpoints}
run_id=${RUN_ID:-qwen0p8_groot_stereo_e5_full_same_row_topk_init_std2e-2_gate1e-1_bidirectional_leftprimary_l40_4gpu_bs8ga4_eff128_seed42_fromscratch_0721_attempt01}
PORT_OVERRIDE=${PORT:-}

if [ "${NUM_PROCESSES}" != "4" ]; then echo "[guard] E5 requires NUM_PROCESSES=4"; exit 3; fi
[ "${FRAMEWORK}" = "QwenGR00T" ] || { echo "[guard] wrong framework"; exit 3; }
[ "${DATA_MIX}" = "libero_all_sfstereo_leftprimary" ] || { echo "[guard] wrong stereo data mix"; exit 3; }
case "${run_id}" in *fromscratch*) ;; *) echo "[guard] run_id must contain fromscratch"; exit 3 ;; esac

for name in RESUME PRETRAINED_CKPT RESUME_FROM_CHECKPOINT RESUME_FROM_CHECKPOINT_DIR CHECKPOINT CHECKPOINT_PATH INIT_FROM; do
  if [ -n "${!name:-}" ]; then echo "[guard] ${name} is forbidden for from-scratch E5"; exit 3; fi
done
[ -z "${FREEZE_MODULES:-}" ] || { echo "[guard] FREEZE_MODULES must be empty"; exit 3; }
[ -f "${config_yaml}" ] || { echo "[preflight] missing ${config_yaml}"; exit 1; }
[ -f "${DS_CONFIG}" ] || { echo "[preflight] missing ${DS_CONFIG}"; exit 1; }
[ -x "${CONDA_VENV}/accelerate" ] || { echo "[preflight] missing accelerate"; exit 1; }

EXPECTED_UUIDS=(
  GPU-abfbb6c8-5cb5-264e-3859-aa6a332b3fb5
  GPU-788fb781-5b80-a333-9b8f-b7e57121e105
  GPU-83104a21-b807-2862-9dcd-e9210fda2223
  GPU-23b0129f-7a1f-a3cc-f4ba-5170697bc9c7
)
TARGET_UUIDS="${EXPECTED_UUIDS[0]},${EXPECTED_UUIDS[1]},${EXPECTED_UUIDS[2]},${EXPECTED_UUIDS[3]}"
E5_FORBIDDEN_UUIDS="GPU-36f8a964-9520-4128-7175-48aec8fd2ebe,GPU-2fb599b9-6174-52dc-d7f4-99a71fc0dd91,GPU-ae578726-e70c-f8ec-6e7c-62c07e64b910,GPU-66adfbd4-8a5a-1ccd-575a-6f608e30265e"

run_root_dir_abs=$(realpath -m "${run_root_dir}")
mkdir -p "${run_root_dir_abs}"
exec 9>"${run_root_dir_abs}/.e5_full_only.lock"
flock -n 9 || { echo "[guard] another E5 launcher holds ${run_root_dir_abs}/.e5_full_only.lock"; exit 3; }

check_target_gpu() {
  local idx="$1" expected="$2" uuid used free util mig apps
  uuid=$(nvidia-smi -i "${idx}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
  used=$(nvidia-smi -i "${idx}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  free=$(nvidia-smi -i "${idx}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
  util=$(nvidia-smi -i "${idx}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
  mig=$(nvidia-smi -i "${idx}" --query-gpu=mig.mode.current --format=csv,noheader,nounits | tr -d '[:space:]')
  apps=$(nvidia-smi -i "${idx}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')
  [ "${uuid}" = "${expected}" ] || { echo "[guard] GPU ${idx} UUID mismatch: ${uuid} != ${expected}"; exit 3; }
  [ -z "${apps}" ] || { echo "[guard] GPU ${idx} has compute process ${apps}"; exit 3; }
  [ "${used}" -le 64 ] || { echo "[guard] GPU ${idx} used memory ${used} MiB"; exit 3; }
  [ "${util}" = "0" ] || { echo "[guard] GPU ${idx} utilization ${util}%"; exit 3; }
  case "${mig}" in Enabled|enabled) echo "[guard] GPU ${idx} MIG enabled"; exit 3 ;; esac
  echo "[gpu-preflight] physical=${idx} uuid=${uuid} free=${free}MiB used=${used}MiB util=${util}% mig=${mig:-unknown} apps=none"
}

echo "[gpu-preflight] checking physical GPU 0-3 before CUDA/model allocation"
for idx in 0 1 2 3; do check_target_gpu "${idx}" "${EXPECTED_UUIDS[${idx}]}"; done
sleep 1
for idx in 0 1 2 3; do check_target_gpu "${idx}" "${EXPECTED_UUIDS[${idx}]}"; done

for idx in 4 5 6 7; do
  apps=$(nvidia-smi -i "${idx}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')
  [ -n "${apps}" ] || { echo "[guard] expected active E3 process missing on physical GPU ${idx}"; exit 3; }
  echo "[gpu-preflight] E3 remains active on physical GPU ${idx}: pids=${apps}"
done

output_dir="${run_root_dir_abs}/${run_id}"
if ! mkdir "${output_dir}"; then
  echo "[guard] attempt directory already exists: ${output_dir}; choose attempt02 or a new RUN_ID"
  exit 3
fi
mkdir "${output_dir}/checkpoints"
cp "$0" "${output_dir}/launcher.sh"

allocate_port() {
  ${CONDA_VENV}/python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

if [ -n "${PORT_OVERRIDE}" ]; then
  PREFLIGHT_PORT="${PORT_OVERRIDE}"
else
  PREFLIGHT_PORT="$(allocate_port)"
fi
[ -n "${PREFLIGHT_PORT}" ] || { echo "[guard] failed to allocate preflight rendezvous port"; exit 3; }

echo "[gpu-preflight] distributed rank/UUID/PCI check before model/data initialization"
  CUDA_VISIBLE_DEVICES="${TARGET_UUIDS}" \
  E5_TARGET_UUIDS="${TARGET_UUIDS}" \
  E5_FORBIDDEN_UUIDS="${E5_FORBIDDEN_UUIDS}" \
  "${CONDA_VENV}/accelerate" launch \
  --num_processes 4 \
  --main_process_port "${PREFLIGHT_PORT}" \
  scripts/a800/preflight_e5_gpu_assignment.py \
  --expected-uuids "${TARGET_UUIDS}" \
  --forbidden-uuids "${E5_FORBIDDEN_UUIDS}"

if [ -n "${PORT_OVERRIDE}" ]; then
  PORT="${PORT_OVERRIDE}"
else
  PORT="$(allocate_port)"
fi
[ -n "${PORT}" ] || { echo "[guard] failed to allocate training rendezvous port"; exit 3; }

exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "[launch] E5-full-only; E3 layers=3,7,11,15,19,23; E5 layers=11,15,19,23"
echo "[launch] CUDA_VISIBLE_DEVICES=${TARGET_UUIDS} physical GPU set=0,1,2,3 port=${PORT}"
echo "[launch] run_id=${run_id} bs=${BS} processes=${NUM_PROCESSES} GA=4 effective_batch=$((BS*NUM_PROCESSES*4)) max_steps=${MAX_STEPS}"
echo "[launch] from-scratch: no pretrained checkpoint, no resume, no auto-eval"
{
  echo "run_id=${run_id}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_status=$(git status --short | tr '\n' ';')"
  echo "port=${PORT}"
  echo "target_gpu_uuids=${TARGET_UUIDS}"
  echo "data_mix=${DATA_MIX}"
  echo "data_root=${DATA_ROOT}"
  echo "e3_init_std=0.02"
  echo "e5_layers=11,15,19,23"
  echo "e5_topk=4"
  echo "e5_gate_init=0.1"
  echo "max_steps=${MAX_STEPS} warmup=${WARMUP_STEPS} save_interval=${SAVE_INTERVAL}"
} > "${output_dir}/launch_provenance.env"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[dry-run] preflight and provenance succeeded; training not started"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${TARGET_UUIDS}" "${CONDA_VENV}/accelerate" launch \
  --config_file "${DS_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${FRAMEWORK}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.qwenvl.stereo_cam_rope_enabled false \
  --framework.qwenvl.stereo_cam_rope_num_cameras 2 \
  --framework.qwenvl.stereo_cam_rope_baseline_m 0.06 \
  --framework.qwenvl.stereo_cam_rope_fovy_degrees 45.0 \
  --framework.qwenvl.stereo_cam_rope_image_width 256 \
  --framework.qwenvl.stereo_cam_rope_image_height 256 \
  --framework.qwenvl.stereo_cam_rope_spatial_merge 2 \
  --framework.qwenvl.stereo_cam_rope_right_first true \
  --framework.qwenvl.stereo_cam_branch_enabled true \
  --framework.qwenvl.stereo_cam_branch_heads 4 \
  --framework.qwenvl.stereo_cam_branch_head_dim 128 \
  --framework.qwenvl.stereo_cam_branch_layers 3,7,11,15,19,23 \
  --framework.qwenvl.stereo_cam_branch_out_proj_init_std 0.02 \
  --framework.qwenvl.stereo_cam_branch_e5_enabled true \
  --framework.qwenvl.stereo_cam_branch_e5_layers 11,15,19,23 \
  --framework.qwenvl.stereo_cam_branch_e5_topk 4 \
  --framework.qwenvl.stereo_cam_branch_e5_gate_init 0.1 \
  --framework.qwenvl.stereo_cam_branch_e5_out_proj_init_std 0.02 \
  --framework.qwenvl.stereo_cam_branch_e5_init_seed 42 \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${BS}" \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.pretrained_checkpoint "" \
  --trainer.freeze_modules "" \
  --trainer.is_resume false \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.num_warmup_steps "${WARMUP_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 999999 \
  --run_root_dir "${run_root_dir_abs}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Qwen0p8_GR00T_Stereo \
  --wandb_entity disabled
