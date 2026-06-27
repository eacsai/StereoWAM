#!/usr/bin/env bash
# Full LEFTPRIMARY cache rebuild on 4090d. Fans 4 FFS-net0 + 4 Utonia-perpatch suite jobs
# concurrently across idle GPUs 0/1/4/5 (2 jobs/GPU). Thread-capped good-neighbour: the box
# is shared (arm* trainings on GPUs 2/3/6). Each job logs to its own per-suite log under
# playground/Checkpoints/. --resume safe (re-run picks up partial). Run in detached tmux.
# Outputs:
#   FFS    -> playground/Datasets/ffs_net0_cache_leftprimary_4090d/<suite>/
#   Utonia -> playground/Datasets/utonia_cache_perpatch_leftprimary_4090d/<suite>/
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
exec > playground/Checkpoints/precompute_lp_master.log 2>&1
DRV=scripts/4090d
GPUS=(0 1 4 5)
# NOTE: array MUST NOT be named SUITES — the FFS wrapper reads a SUITES env var; the
# `SUITES=...` env prefix would clobber the array and break ${SUITES[$i]} for i>0 under set -u.
SUITE_LIST=(libero_object_no_noops_1.0.0_lerobot libero_goal_no_noops_1.0.0_lerobot \
            libero_spatial_no_noops_1.0.0_lerobot libero_10_no_noops_1.0.0_lerobot)

echo "===== START leftprimary precompute (4 FFS + 4 Utonia, 8 concurrent) $(date -u +%FT%TZ) ====="
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} NPROC_PER_WORKER=16 SUITES="${SUITE_LIST[$i]}" \
    LOGNAME="ffs_precompute_${SUITE_LIST[$i]}" bash "$DRV/run_precompute_ffs_4090d.sh" &
done
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} NPROC_PER_WORKER=16 \
    LOGNAME="utonia_precompute_${SUITE_LIST[$i]}" bash "$DRV/run_precompute_utonia_suite_4090d.sh" "${SUITE_LIST[$i]}" &
done
wait
echo "===== ALL DONE $(date -u +%FT%TZ) ====="
echo "ALL_PRECOMPUTE_LEFTPRIMARY_DONE"
