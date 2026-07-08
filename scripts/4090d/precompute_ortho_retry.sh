#!/usr/bin/env bash
# Auto-retry wrapper for ortho-view cache precompute on the shared 4090d host.
#
# Why this exists: the host (1TB RAM, shared with other tenants) periodically
# runs low on RAM; PyAV video decoding in the dataloader then dies with
# `av.error.MemoryError: [Errno 12] Cannot allocate memory`. This is an
# EXTERNAL transient (neighbor memory pressure), not a bug in our render/
# dataloader code (see memory feedback_stagger_compile_heavy_trainings).
# --resume skips already-written rows via done.npy, so a restart continues
# from the last checkpoint with no lost work and no overwrite.
#
# Loops the precompute until the suite prints ORTHO_PRECOMPUTE_DONE, then
# exits 0. Keeping this wrapper alive inside the tmux session means the tmux
# stays ALIVE across python crashes, so the completion/crash monitor only sees
# a real crash if the wrapper itself dies.
set -u
SUITE="$1"      # e.g. libero_object_no_noops_1.0.0_lerobot
GPU="$2"        # e.g. 1
LOGTAG="$3"     # e.g. libero_obj  -> tmp/precompute_<tag>.log
REPO=/data/wangqiwei/ICLR2026/starVLA
CACHE_DIR=playground/Caches/ortho_views_leftprimary_probe_v1
LOG="tmp/precompute_${LOGTAG}.log"
cd "$REPO" || exit 1

attempt=0
while true; do
  if grep -aq ORTHO_PRECOMPUTE_DONE "$LOG" 2>/dev/null; then
    echo "WRAPPER: $SUITE already marked DONE, exiting 0" | tee -a "$LOG"; exit 0
  fi
  attempt=$((attempt+1))
  echo "WRAPPER: === attempt $attempt for $SUITE on GPU$GPU at $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
    OPENBLAS_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24 \
    .venv/bin/python scripts/4090d/precompute_ortho_view_cache.py \
      --cache-dir "$CACHE_DIR" --suites "$SUITE" >> "$LOG" 2>&1
  rc=$?
  if grep -aq ORTHO_PRECOMPUTE_DONE "$LOG" 2>/dev/null; then
    echo "WRAPPER: $SUITE DONE after attempt $attempt (rc=$rc)" | tee -a "$LOG"; exit 0
  fi
  echo "WRAPPER: $SUITE exited rc=$rc WITHOUT done marker; retrying in 90s (--resume continues)" | tee -a "$LOG"
  sleep 90
done
