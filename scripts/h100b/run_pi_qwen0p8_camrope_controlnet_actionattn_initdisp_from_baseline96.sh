#!/bin/bash
# H100b launcher: PI + Qwen3.5-0.8B + cam_rope + FFS ControlVLA-style branch (V4).
#
# V4 vs V1 (run_pi_qwen0p8_camrope_epipolar_ffs_controlnet_fromscratch.sh):
#   * framework.name           QwenPIControlNetFFS -> QwenPIControlVLAFFS
#   * config tree              ffs_controlnet.*    -> ffs_controlvla.*
#   * ffs_pool_size            (new — 8x8 = 64 FFS tokens for SDPA K_z/V_z)
#   * stereo_epipolar_mask     dropped (abandoned, see memory project_drop_epipolar_decision)
#   * everything else identical (eff_batch 96, 30k steps, fromscratch, same data, same FFS ckpt)
#
# Injection mechanism: per-DiT-layer parallel K_z/V_z inside action cross-attn
# (controlvla_branch.py), trunk attn unchanged, K_z=V_z=0 at step 0.
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

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-pi_qwen0p8_camrope_controlnet_actionattn_initdisp_from_baseline96_h100b_$(date +%m%d)}

BS=${BS:-24}                                     # was 12 in V1 — bump for less GA on H100 80GB
MAX_STEPS=${MAX_STEPS:-30000}
GPUS=${GPUS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
PORT=${PORT:-29703}                              # +1 vs V1 launcher to avoid clash if both running
DS_CONFIG=${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_ga2.yaml}    # GA=2 (eff=96 with BS=24x2)

# CODEX FIX (round-3 HIGH): preflight — verify DS_CONFIG and the inner JSON it
# wraps both exist. Avoids "starts then crashes" when newly-added GA configs
# fall through .gitignore cracks (the wrapper points at an inner JSON file).
if [ ! -f "$DS_CONFIG" ]; then
  echo "[preflight] FATAL: DS_CONFIG=$DS_CONFIG does not exist"; exit 1
fi
inner_json=$(grep -oE 'deepspeed_config_file: *"?[^"]+' "$DS_CONFIG" | sed 's|.*: *"\?||')
if [ -n "$inner_json" ] && [ ! -f "$inner_json" ]; then
  echo "[preflight] FATAL: DS_CONFIG wraps inner $inner_json which does not exist"; exit 1
fi
echo "[preflight] DS_CONFIG=$DS_CONFIG + inner=$inner_json — both present"

# Symlink stereo dataset (fail-closed, same as V1)
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

echo "[launch] PI + Qwen3.5-0.8B + cam_rope + ControlVLA-style branch (FFS 20-30-48 frozen, no epipolar)"
echo "[launch] BS=$BS x $NUM_PROCESSES GPU x GA=2 = eff_batch=96 (matches 4090d 6-GPU eff=96)"
echo "[launch] DS_CONFIG=$DS_CONFIG — if OOM, fall back BS=12 GA=4 via env override"

CUDA_VISIBLE_DEVICES=${GPUS} ${CONDA_VENV}/accelerate launch \
  --config_file ${DS_CONFIG} \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name "QwenPIControlVLAFFS" \
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
  --trainer.pretrained_checkpoint /mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints/goal_phase3b_camrope_0523_baseline_for_initdisp_init/checkpoints/steps_30000_pytorch_model.pt \
  --framework.ffs_controlvla.ffs_model_path ${ffs_model_path} \
  --framework.ffs_controlvla.ffs_scale 0 \
  --framework.ffs_controlvla.ffs_image_size 256 \
  --framework.ffs_controlvla.ffs_pool_size 8 \
  --framework.ffs_controlvla.use_init_disp true \
  --framework.ffs_controlvla.ffs_expected_sha256 98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692 \
  --framework.ffs_controlvla.primary_idx 0 \
  --framework.ffs_controlvla.right_view_idx 1 \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size $BS \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps $MAX_STEPS \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 999999 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_FFS_ControlVLA \
  --wandb_entity disabled \
  "${RESUME_FLAG[@]}"
