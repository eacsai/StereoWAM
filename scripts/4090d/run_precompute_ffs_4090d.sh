#!/usr/bin/env bash
# FFS net0 LEFTPRIMARY cache precompute on 4090d. Single process (FFS is fast, ~tens of min).
# 4090d precompute wrapper (Python impl in scripts/tools/): .venv (not /opt/conda),
# /data paths, FFS_REPO_DIR on 4090d. Good-neighbour: CPU thread cap + nice/ionice
# (4090d is shared; other arm* trainings run concurrently). The precompute script itself
# is leftprimary-scoped (DEFAULT_DATA_MIX=leftprimary + hard guard).
# Env knobs: CACHE_DIR SUITES LIMIT_ROWS CUDA_VISIBLE_DEVICES LOGNAME NPROC_PER_WORKER.
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
CACHE_DIR="${CACHE_DIR:-playground/Datasets/ffs_net0_cache_leftprimary_4090d}"
SUITES="${SUITES:-all}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
LOGNAME="${LOGNAME:-ffs_precompute_lp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FFS_REPO_DIR=/data/wangqiwei/ICLR2026/Fast-FoundationStereo
NPROC="${NPROC_PER_WORKER:-32}"
export OMP_NUM_THREADS=$NPROC MKL_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC \
       NUMEXPR_NUM_THREADS=$NPROC VECLIB_MAXIMUM_THREADS=$NPROC
exec > "playground/Checkpoints/${LOGNAME}.log" 2>&1
echo "[ffs precompute] start $(date -u +%Y-%m-%dT%H:%M:%SZ) cache=$CACHE_DIR suites=$SUITES limit=$LIMIT_ROWS gpu=$CUDA_VISIBLE_DEVICES nproc=$NPROC"
nice -n 10 ionice -c2 -n5 .venv/bin/python scripts/tools/precompute_ffs_net0_cache.py \
  --cache-dir "$CACHE_DIR" \
  --suites "$SUITES" \
  --data-root playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW \
  --data-mix libero_all_sfstereo_leftprimary \
  --ffs-model-path /data/wangqiwei/ICLR2026/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth \
  --ffs-expected-sha256 98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692 \
  --limit-rows "$LIMIT_ROWS"
echo "FFS_PRECOMPUTE_EXIT=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
