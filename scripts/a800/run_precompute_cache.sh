#!/usr/bin/env bash
# Full Utonia Version-A offline-cache precompute (4 LIBERO stereo suites -> ~49GB fp16).
# Runs on h100b GPU1 (freed from the stopped random perpatch). FRESH cache dir (no resume
# across config changes — avoids the H1 stale-resume footgun). Launched in a detached tmux.
set -uo pipefail
cd /home/wangqiwei/ICLR2026/starVLA
# Redirect everything to the log so the tmux launch command needs no quoting/redirect.
exec > playground/Checkpoints/utonia_precompute.log 2>&1
export CUDA_VISIBLE_DEVICES=1
export FFS_REPO_DIR=/home/wangqiwei/ICLR2026/Fast-FoundationStereo

echo "[precompute] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[precompute] df before:"
df -h playground/Datasets | tail -2

/home/wangqiwei/ICLR2026/starVLA/.venv/bin/python scripts/a800/precompute_utonia_cache.py \
  --cache-dir playground/Datasets/utonia_cache_perpatch \
  --suites all \
  --data-root playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW \
  --data-mix libero_all_sfstereo_rightprimary \
  --ffs-model-path /home/wangqiwei/ICLR2026/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth \
  --ffs-expected-sha256 98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692 \
  --utonia-ckpt-path ./playground/Pretrained_models/Utonia/utonia.pth \
  --utonia-scale 4.0 \
  --image-size 256 \
  --backproject-stride 4 \
  --fovy-degrees 45.0 \
  --baseline-m 0.06 \
  --depth-min 0.05 \
  --depth-max 3.0 \
  --disp-eps 0.001
echo "PRECOMPUTE_EXIT=$?"
echo "[precompute] df after:"
df -h playground/Datasets | tail -2
