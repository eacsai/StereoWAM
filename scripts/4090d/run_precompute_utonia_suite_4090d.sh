#!/usr/bin/env bash
# Per-suite Utonia LEFTPRIMARY cache precompute on 4090d (batch=1; run the 4 suites in
# parallel on idle GPUs to fill them — single process is ~12% util, CPU/IO-bound).
# Each process writes ONLY its suite subdir -> no shared-file collision. --resume picks up
# partial progress. 4090d-equivalent of scripts/h100b/run_precompute_suite.sh: .venv,
# /data paths, FFS_REPO_DIR on 4090d. Good-neighbour CPU thread cap + nice/ionice.
# SUITE = $1. Env knobs: CACHE_DIR LIMIT_ROWS CUDA_VISIBLE_DEVICES LOGNAME NPROC_PER_WORKER.
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
SUITE="${1:?usage: $0 <suite: object|goal|spatial|10>}"
CACHE_DIR="${CACHE_DIR:-playground/Datasets/utonia_cache_perpatch_leftprimary_4090d}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
LOGNAME="${LOGNAME:-utonia_precompute_${SUITE}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FFS_REPO_DIR=/data/wangqiwei/ICLR2026/Fast-FoundationStereo
NPROC="${NPROC_PER_WORKER:-32}"
export OMP_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC \
       NUMEXPR_NUM_THREADS=$NPROC VECLIB_MAXIMUM_THREADS=$NPROC
exec > "playground/Checkpoints/${LOGNAME}.log" 2>&1
echo "[utonia precompute ${SUITE}] start $(date -u +%Y-%m-%dT%H:%M:%SZ) cache=$CACHE_DIR limit=$LIMIT_ROWS gpu=$CUDA_VISIBLE_DEVICES nproc=$NPROC"
nice -n 10 ionice -c2 -n5 .venv/bin/python scripts/h100b/precompute_utonia_cache.py \
  --cache-dir "$CACHE_DIR" \
  --suites "$SUITE" \
  --data-root playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW \
  --data-mix libero_all_sfstereo_leftprimary \
  --ffs-model-path /data/wangqiwei/ICLR2026/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth \
  --ffs-expected-sha256 98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692 \
  --utonia-ckpt-path ./playground/Pretrained_models/Utonia/utonia.pth \
  --utonia-scale 4.0 --image-size 256 --backproject-stride 4 \
  --fovy-degrees 45.0 --baseline-m 0.06 --depth-min 0.05 --depth-max 3.0 --disp-eps 0.001 \
  --limit-rows "$LIMIT_ROWS"
echo "UTONIA_PRECOMPUTE_EXIT=$? suite=${SUITE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
