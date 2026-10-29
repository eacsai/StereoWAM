#!/usr/bin/env bash
# Matched visual-attention ablation for pure leftprimary stereo.
# MODE=causal      -> Qwen image-image causal + causal Camera Branch.
# MODE=image_bidir -> Qwen image-image bidirectional + bidirectional Camera Branch.
# This launcher is fail-closed and never kills/resets another user's process.
# Historical 30k is a logged micro-step count; see docs/experiments/20261029_l40_results.md.
set -euo pipefail

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd /home/wangqw/ICLR27/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

CONDA_VENV=${CONDA_VENV:-/home/wangqw/ICLR27/starVLA/.venv/bin}
config_yaml=${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_cotrain_libero.yaml}
base_vlm=${BASE_VLM:-/home/wangqw/ICLR27/starVLA/playground/Pretrained_models/Qwen3.5-0.8B}
data_root=${DATA_ROOT:-/home/wangqw/ICLR27/starVLA/playground/Datasets/LEROBOT_LIBERO_STEREO_LEFTPRIMARY}
data_mix=${DATA_MIX:-libero_all_sfstereo_leftprimary}
run_root_dir=${RUN_ROOT_DIR:-./playground/Checkpoints}

MODE=${MODE:-causal}
GPUS=${GPUS:-4,5,6,7}
export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_PROCESSES=${NUM_PROCESSES:-4}
visible_gpu_count=$("$CONDA_VENV/python" -c 'import torch; print(torch.cuda.device_count())')
[ "$visible_gpu_count" = "$NUM_PROCESSES" ] || {
  echo "[guard] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES exposes $visible_gpu_count GPUs; expected $NUM_PROCESSES" >&2
  exit 3
}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-30000}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
PORT=${PORT:-29782}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml}
ATTN_IMPL=sdpa
CAM_BRANCH_HEADS=${CAM_BRANCH_HEADS:-4}
CAM_BRANCH_HEAD_DIM=${CAM_BRANCH_HEAD_DIM:-128}
CAM_BRANCH_OUT_PROJ_INIT_STD=${CAM_BRANCH_OUT_PROJ_INIT_STD:-0.02}
DRY_RUN=${DRY_RUN:-0}
case "$DRY_RUN" in
  0|1) ;;
  *) echo "[guard] DRY_RUN must be 0 or 1, got '$DRY_RUN'" >&2; exit 3 ;;
esac

case "$MODE" in
  causal)
    BRANCH_CAUSAL=true
    DEFAULT_RUN_ID=qwen0p8_groot_visual_causal_leftprimary_l40_4gpu_bs8ga4_eff128_seed42_fromscratch
    ;;
  image_bidir)
    BRANCH_CAUSAL=false
    DEFAULT_RUN_ID=qwen0p8_groot_visual_image_bidir_leftprimary_l40_4gpu_bs8ga4_eff128_seed42_fromscratch
    ;;
  *)
    echo "[guard] MODE must be causal or image_bidir, got '$MODE'" >&2
    exit 3
    ;;
esac
RUN_ID=${RUN_ID:-$DEFAULT_RUN_ID}

for required in "$DS_CONFIG" "$config_yaml"; do
  [ -f "$required" ] || { echo "[preflight] missing file: $required" >&2; exit 1; }
done
[ "$PER_DEVICE_BATCH" = "8" ] || { echo "[guard] PER_DEVICE_BATCH must be 8 for the approved run" >&2; exit 3; }
[ "$GRAD_ACCUM" = "4" ] || { echo "[guard] GRAD_ACCUM must be 4 for the approved DeepSpeed config" >&2; exit 3; }
[ "$NUM_PROCESSES" = "4" ] || { echo "[guard] NUM_PROCESSES must be 4 for the approved run" >&2; exit 3; }
[ "$DS_CONFIG" = "starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml" ] || { echo "[guard] DS_CONFIG must be the approved ga4 config" >&2; exit 3; }
ds_config_detail=starVLA/config/deepseeds/ds_config_ga4.yaml
[ -f "$ds_config_detail" ] || { echo "[preflight] missing DeepSpeed detail config: $ds_config_detail" >&2; exit 1; }
grep -Eq '"gradient_accumulation_steps"[[:space:]]*:[[:space:]]*4([,[:space:]]|$)' "$ds_config_detail" || { echo "[guard] DeepSpeed detail config must set gradient_accumulation_steps=4" >&2; exit 3; }
[ -d "$base_vlm" ] || { echo "[preflight] missing base VLM: $base_vlm" >&2; exit 1; }
[ -d "$data_root" ] || { echo "[preflight] missing data root: $data_root" >&2; exit 1; }
[ "$data_mix" = "libero_all_sfstereo_leftprimary" ] || {
  echo "[guard] this launcher requires leftprimary data mix, got '$data_mix'" >&2
  exit 3
}

output_dir=${run_root_dir}/${RUN_ID}
[ ! -e "$output_dir" ] || {
  echo "[guard] output already exists; refusing resume/overwrite: $output_dir" >&2
  exit 3
}

preflight=(
  "$CONDA_VENV/python" scripts/a800/preflight_visual_attention_ablation.py
  --gpus "$GPUS"
  --mode "$MODE"
  --branch-causal "$BRANCH_CAUSAL"
  --attention-implementation "$ATTN_IMPL"
  --per-device-batch "$PER_DEVICE_BATCH"
  --grad-accum "$GRAD_ACCUM"
  --world-size "$NUM_PROCESSES"
  --expected-effective-batch 128
  --expected-per-device-batch 8
  --expected-grad-accum 4
  --expected-world-size 4
  --check-gpus
)
"${preflight[@]}"

lock_root=/tmp/starvla_visual_attention_gpu_locks
mkdir -p "$lock_root"
IFS=',' read -r -a gpu_list <<< "$GPUS"
lock_paths=()
cleanup_locks() { for path in "${lock_paths[@]}"; do rmdir "$path" 2>/dev/null || true; done; }
trap cleanup_locks EXIT
for gpu in "${gpu_list[@]}"; do
  lock_path="$lock_root/gpu_$gpu"
  if ! mkdir "$lock_path" 2>/dev/null; then
    cleanup_locks
    echo "[guard] target GPU reservation already exists: $lock_path" >&2
    exit 3
  fi
  lock_paths+=("$lock_path")
done
# Recheck after reservation: no kill/reset and no stale preflight assumption.
"${preflight[@]}"

cmd=(
  "$CONDA_VENV/accelerate" launch
  --config_file "$DS_CONFIG"
  --num_processes "$NUM_PROCESSES"
  # Legacy baseline-compatible: DeepSpeed JSON owns GA=4; Accelerate keeps GA=1.
  --main_process_port "$PORT"
  starVLA/training/train_starvla.py
  --config_yaml "$config_yaml"
  --framework.name QwenGR00T
  --framework.qwenvl.base_vlm "$base_vlm"
  --framework.qwenvl.attn_implementation "$ATTN_IMPL"
  --framework.qwenvl.stereo_cam_branch_enabled true
  --framework.qwenvl.stereo_visual_attention_mode "$MODE"
  --framework.qwenvl.stereo_cam_branch_is_causal "$BRANCH_CAUSAL"
  --framework.qwenvl.stereo_cam_branch_heads "$CAM_BRANCH_HEADS"
  --framework.qwenvl.stereo_cam_branch_head_dim "$CAM_BRANCH_HEAD_DIM"
  --framework.qwenvl.stereo_cam_branch_out_proj_init_std "$CAM_BRANCH_OUT_PROJ_INIT_STD"
  --framework.qwenvl.stereo_cam_rope_num_cameras 2
  --framework.qwenvl.stereo_cam_rope_baseline_m 0.06
  --framework.qwenvl.stereo_cam_rope_fovy_degrees 45.0
  --framework.qwenvl.stereo_cam_rope_image_width 256
  --framework.qwenvl.stereo_cam_rope_image_height 256
  --framework.qwenvl.stereo_cam_rope_spatial_merge 2
  --framework.qwenvl.stereo_cam_rope_right_first true
  --framework.qwenvl.stereo_epipolar_mask_enabled false
  --datasets.vla_data.data_root_dir "$data_root"
  --datasets.vla_data.data_mix "$data_mix"
  --datasets.vla_data.per_device_batch_size "$PER_DEVICE_BATCH"
  --trainer.vla_data.video_backend torchvision_av
  --trainer.freeze_modules ""
  --trainer.max_train_steps "$MAX_STEPS"
  --trainer.num_warmup_steps "$WARMUP_STEPS"
  --trainer.save_interval "$SAVE_INTERVAL"
  --trainer.logging_frequency 20
  --trainer.eval_interval 999999
  --run_root_dir "$run_root_dir"
  --run_id "$RUN_ID"
  --wandb_project starVLA_Qwen0p8_GR00T_visual_attention
  --wandb_entity disabled
)

if [ "$DRY_RUN" = "1" ]; then
  printf '[dry-run] CUDA_VISIBLE_DEVICES=%s ' "$GPUS"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  echo "[dry-run] mode=$MODE branch_causal=$BRANCH_CAUSAL attention=$ATTN_IMPL effective_batch=128"
  exit 0
fi

mkdir -p "$run_root_dir"
mkdir "$output_dir"
cp "$0" "$output_dir/"
provenance_files=(
  starVLA/model/framework/VLM4A/QwenGR00T.py
  starVLA/model/modules/stereo/cam_branch_attention.py
  starVLA/model/modules/stereo/visual_attention.py
  scripts/a800/preflight_visual_attention_ablation.py
  scripts/a800/run_qwen0p8_groot_stereo_visual_attention_l40.sh
  scripts/a800/smoke_visual_attention_gpu.py
  scripts/a800/test_visual_attention_ablation.py
  "$config_yaml"
  "$DS_CONFIG"
  "$ds_config_detail"
)
mkdir -p "$output_dir/provenance_files"
for source_file in "${provenance_files[@]}"; do
  [ -f "$source_file" ] || { echo "[provenance] missing source file: $source_file" >&2; exit 1; }
  cp -p "$source_file" "$output_dir/provenance_files/$(basename "$source_file")"
done
{
  echo "mode=$MODE"
  echo "branch_causal=$BRANCH_CAUSAL"
  echo "attention_implementation=$ATTN_IMPL"
  echo "gpus=$GPUS"
  echo "per_device_batch=$PER_DEVICE_BATCH"
  echo "grad_accum=$GRAD_ACCUM"
  echo "accelerate_gradient_accumulation_steps=implicit_default_1"
  echo "effective_batch=128"
  echo "git_head=$(git rev-parse HEAD)"
  echo "tracked_diff_sha256=$(git diff --binary HEAD | sha256sum | awk '{print $1}')"
  echo "deepspeed_detail_config=$ds_config_detail"
  echo "implementation_file_hashes:"
  for source_file in "${provenance_files[@]}"; do
    sha256sum "$source_file"
  done
  echo "package_versions=$("$CONDA_VENV/python" -c 'import importlib.metadata as md, torch, transformers; print("torch={} transformers={} accelerate={} deepspeed={}".format(torch.__version__, transformers.__version__, md.version("accelerate"), md.version("deepspeed")))')"
  echo "status:"
  git status --porcelain=v1
  echo "command:"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$output_dir/provenance.txt"

echo "[launch] fixed GPU set=$GPUS mode=$MODE branch_causal=$BRANCH_CAUSAL effective_batch=128 steps=$MAX_STEPS"
CUDA_VISIBLE_DEVICES="$GPUS" "${cmd[@]}"
