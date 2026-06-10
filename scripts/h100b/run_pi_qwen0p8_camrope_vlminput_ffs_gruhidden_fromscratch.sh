#!/bin/bash
# H100b launcher: PI + Qwen3.5-0.8B + cam_rope + FFS injected at the VLM INPUT
# (zero-gated ControlNet-style residual on the merged inputs_embeds, primary-view
# image tokens). Feature = FFS GRU-HIDDEN net[0] (post-cost-volume, disparity-aware).
#
# vs run_pi_qwen0p8_camrope_controlvla_gruhidden_fromscratch_90k.sh:
#   * framework = QwenPIVLMInputFFS (NEW) instead of QwenPIControlVLAFFS.
#     - ControlVLA injects a parallel K/V branch in the ACTION DiT cross-attn.
#     - VLMInput injects into the VLM's merged inputs_embeds so the stereo signal
#       flows through the WHOLE VLM and fuses with the language instruction
#       (cf. StereoPolicy / StereoVLA; our differentiator = zero-gated ControlNet).
#   * config keys live under framework.ffs_vlm_input.* (not ffs_controlvla.*).
#   * 30k (not 90k): 90k convergence test already showed gru_hidden plateaus ~0.90
#     on libero_goal; 30k is enough to read this variant's converged number.
#   * fromscratch (NO warm-start), same eff_batch 96 / data / FFS ckpt / cam_rope.
#
# Anti-drift: the injector is gate*proj(ffs) with a SINGLE scalar gate init 0
# (spatial_proj normal-init) -> step-0 byte-identical VLM, slow global ramp.
set -e
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
# Pin FFS ckpt hash (same file on h100b /mnt/data and 4090d /data).
ffs_sha256=98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-pi_qwen0p8_camrope_vlminput_ffs_gruhidden_fromscratch_h100b_$(date +%m%d)}

BS=${BS:-24}                                     # eff_batch = BS x NUM_PROCESSES x GA(2) = 96
MAX_STEPS=${MAX_STEPS:-30000}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29707}                              # distinct from 90k(29706)/controlvla(29701)
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga2.yaml}    # GA=2 (eff=96 with BS=24x2)

# Preflight: verify DS_CONFIG and the inner JSON it wraps both exist.
if [ ! -f "$DS_CONFIG" ]; then
  echo "[preflight] FATAL: DS_CONFIG=$DS_CONFIG does not exist"; exit 1
fi
inner_json=$(grep -oE 'deepspeed_config_file: *"?[^"]+' "$DS_CONFIG" | sed 's|.*: *"\?||')
if [ -n "$inner_json" ] && [ ! -f "$inner_json" ]; then
  echo "[preflight] FATAL: DS_CONFIG wraps inner $inner_json which does not exist"; exit 1
fi
echo "[preflight] DS_CONFIG=$DS_CONFIG + inner=$inner_json — both present"

# Symlink stereo dataset (fail-closed, same as the ControlVLA variant)
STEREO_TARGET=/mnt/data/wangqiwei/wangqiwei/libero_goal_openvla_vanilla_stereo_lerobot
mkdir -p $libero_data_root
LINK_PATH=$libero_data_root/libero_goal
if [ -e "$LINK_PATH" ] || [ -L "$LINK_PATH" ]; then
  if [ -L "$LINK_PATH" ]; then
    current_target=$(readlink -f "$LINK_PATH")
    expected_target=$(readlink -f "$STEREO_TARGET")
    if [ "$current_target" = "$expected_target" ]; then
      echo "[symlink] $LINK_PATH already points to $STEREO_TARGET — ok"
    else
      echo "[symlink] replacing stale symlink at $LINK_PATH (was $current_target, want $STEREO_TARGET)"
      rm -f "$LINK_PATH"
      ln -s "$STEREO_TARGET" "$LINK_PATH"
    fi
  else
    echo "[symlink] FAIL-CLOSED: $LINK_PATH exists but is a real directory/file, not a symlink."
    echo "[symlink] please manually remove or fix: $LINK_PATH"
    exit 1
  fi
else
  ln -s "$STEREO_TARGET" "$LINK_PATH"
  echo "[symlink] created $LINK_PATH -> $STEREO_TARGET"
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
mkdir -p ${output_dir}/checkpoints
cp "$0" "${output_dir}/"

RESUME_FLAG=()
shopt -s nullglob
existing_ckpts=("${output_dir}/checkpoints"/steps_*_pytorch_model.pt)
shopt -u nullglob
if [ "${#existing_ckpts[@]}" -gt 0 ]; then
  [ "${RESUME:-0}" = "1" ] && RESUME_FLAG=(--trainer.is_resume true) || { echo "refuse: ckpts present, set RESUME=1"; exit 1; }
fi

echo "[launch] PI + Qwen3.5-0.8B + cam_rope + VLM-INPUT FFS injection, feature_source=GRU_HIDDEN net[0] (frozen FFS, no epipolar, fromscratch, 30k)"
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA=2 = eff_batch=96"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name "QwenPIVLMInputFFS" \
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
  --framework.action_model.diffusion_model_cfg.interleave_self_attention false \
  --framework.ffs_vlm_input.ffs_model_path ${ffs_model_path} \
  --framework.ffs_vlm_input.ffs_expected_sha256 ${ffs_sha256} \
  --framework.ffs_vlm_input.ffs_feature_source gru_hidden \
  --framework.ffs_vlm_input.gru_hidden_dim 16 \
  --framework.ffs_vlm_input.ffs_scale 0 \
  --framework.ffs_vlm_input.ffs_image_size 256 \
  --framework.ffs_vlm_input.inject_hidden_dim 256 \
  --framework.ffs_vlm_input.num_cameras 2 \
  --framework.ffs_vlm_input.primary_idx 0 \
  --framework.ffs_vlm_input.right_view_idx 1 \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_FFS_VLMInput \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
