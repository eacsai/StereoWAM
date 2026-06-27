#!/usr/bin/env bash
# Per-suite Utonia precompute (batch=1, one process per suite -> run 4 in parallel on the
# same GPU to fill the idle GPU [single process is ~12% util, CPU-bound on transform/IO]).
# Each process writes ONLY its own suite subdir -> no shared-file/memmap collision.
# batch=1 is mandatory (Utonia forward is NOT batch-invariant); parallelism, not batching,
# is how we use the spare GPU/CPU. --resume picks up any partial progress (done.npy).
set -uo pipefail
cd /home/wangqiwei/ICLR2026/starVLA
SUITE="$1"
exec > "playground/Checkpoints/utonia_precompute_${SUITE}.log" 2>&1
export CUDA_VISIBLE_DEVICES=1
export FFS_REPO_DIR=/home/wangqiwei/ICLR2026/Fast-FoundationStereo
# Cap per-process CPU threads so N parallel workers don't oversubscribe the 192 cores
# (each process's torch/numpy/spconv default to all-cores -> N processes thrash). Tunable.
NPROC_PER_WORKER=${NPROC_PER_WORKER:-24}
export OMP_NUM_THREADS=$NPROC_PER_WORKER MKL_NUM_THREADS=$NPROC_PER_WORKER \
       OPENBLAS_NUM_THREADS=$NPROC_PER_WORKER NUMEXPR_NUM_THREADS=$NPROC_PER_WORKER \
       VECLIB_MAXIMUM_THREADS=$NPROC_PER_WORKER
echo "[precompute ${SUITE}] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
/home/wangqiwei/ICLR2026/starVLA/.venv/bin/python scripts/a800/precompute_utonia_cache.py \
  --cache-dir playground/Datasets/utonia_cache_perpatch \
  --suites "${SUITE}" \
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
echo "PRECOMPUTE_SUITE_EXIT=$? suite=${SUITE} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
