#!/bin/bash
set -euo pipefail

run_id=${RUN_ID:-pi_qwen0p8_llama_adapter_ffs_warmstart_$(date +%m%d)}
PORT=${PORT:-29734}
FREEZE_MODULES=${FREEZE_MODULES:-qwen_vl_interface}
RUN_DESC="LLaMA-Adapter FFS: frozen VLM warm-start, train action head + shared hint + 6 zero-conv gates"

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FFS_REPO_DIR=/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo

cd /mnt/data/wangqiwei/wangqiwei/starVLA
export PYTHONPATH=$(pwd):${FFS_REPO_DIR}:${PYTHONPATH:-}

CONDA_VENV=/opt/conda/envs/starvla/bin
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_STEREO_DATA
data_mix=libero_goal_stereo
base_vlm=./playground/Pretrained_models/Qwen3.5-0.8B
ffs_model_path=${FFS_REPO_DIR}/weights/20-30-48/model_best_bp2_serialize.pth
ffs_sha256=98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692
pretrained_ckpt=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints/goal_phase3b_camrope_0523/checkpoints/steps_30000_pytorch_model.pt

run_root_dir=./playground/Checkpoints
BS=${BS:-24}
MAX_STEPS=${MAX_STEPS:-30000}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga2.yaml}

if [ ! -f "$pretrained_ckpt" ]; then
  echo "[preflight] FATAL: pretrained_ckpt=$pretrained_ckpt does not exist; confirm the h100b mirror path"; exit 1
fi
if [ ! -f "$ffs_model_path" ]; then
  echo "[preflight] FATAL: ffs_model_path=$ffs_model_path does not exist"; exit 1
fi
if [ ! -f "$DS_CONFIG" ]; then
  echo "[preflight] FATAL: DS_CONFIG=$DS_CONFIG does not exist"; exit 1
fi
inner_json=$(grep -oE 'deepspeed_config_file: *"?[^"]+' "$DS_CONFIG" | sed -E 's|.*: *"?||' || true)
if [ -n "$inner_json" ] && [ ! -f "$inner_json" ]; then
  echo "[preflight] FATAL: DS_CONFIG wraps inner $inner_json which does not exist"; exit 1
fi
echo "[preflight] DS_CONFIG=$DS_CONFIG inner=${inner_json:-none}"

STEREO_TARGET=/mnt/data/wangqiwei/wangqiwei/libero_goal_openvla_vanilla_stereo_lerobot
mkdir -p "$libero_data_root"
LINK_PATH=$libero_data_root/libero_goal
if [ -e "$LINK_PATH" ] || [ -L "$LINK_PATH" ]; then
  if [ -L "$LINK_PATH" ]; then
    current_target=$(readlink -f "$LINK_PATH")
    expected_target=$(readlink -f "$STEREO_TARGET")
    if [ "$current_target" != "$expected_target" ]; then
      echo "[symlink] replacing stale symlink at $LINK_PATH (was $current_target, want $STEREO_TARGET)"
      rm -f "$LINK_PATH"
      ln -s "$STEREO_TARGET" "$LINK_PATH"
    fi
  else
    echo "[symlink] FAIL-CLOSED: $LINK_PATH exists but is not a symlink"; exit 1
  fi
else
  ln -s "$STEREO_TARGET" "$LINK_PATH"
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}/checkpoints"
cp "$0" "${output_dir}/"

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
if [ "${#existing_ckpts[@]}" -gt 0 ]; then
  if [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG=(--trainer.is_resume true)
  else
    echo "refuse: ckpts present, set RESUME=1"; exit 1
  fi
fi

echo "[launch] ${RUN_DESC}"
echo "[launch] run_id=${run_id} port=${PORT}"
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA=2 = eff_batch=96"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name "QwenPILlamaAdapterFFS" \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.stereo_cam_rope_enabled true \
  --framework.qwenvl.stereo_cam_rope_d_c 16 \
  --framework.qwenvl.stereo_cam_rope_num_cameras 2 \
  --framework.qwenvl.stereo_cam_rope_baseline_m 0.06 \
  --framework.qwenvl.stereo_cam_rope_fovy_degrees 45.0 \
  --framework.qwenvl.stereo_cam_rope_image_width 256 \
  --framework.qwenvl.stereo_cam_rope_image_height 256 \
  --framework.qwenvl.stereo_cam_rope_spatial_merge 2 \
  --framework.qwenvl.stereo_cam_rope_init_mode zero \
  --framework.qwenvl.stereo_epipolar_mask_enabled false \
  --framework.action_model.diffusion_model_cfg.interleave_self_attention true \
  --framework.ffs_llama_adapter.ffs_model_path ${ffs_model_path} \
  --framework.ffs_llama_adapter.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_llama_adapter.ffs_feature_source gru_hidden \
  --framework.ffs_llama_adapter.gru_hidden_dim 16 \
  --framework.ffs_llama_adapter.ffs_image_size 256 \
  --framework.ffs_llama_adapter.primary_idx 0 \
  --framework.ffs_llama_adapter.right_view_idx 1 \
  --framework.ffs_llama_adapter.num_cameras 2 \
  --framework.ffs_llama_adapter.hint_hidden_dim 256 \
  --framework.ffs_llama_adapter.reverse_image_order true \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_FFS_LlamaAdapter \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
