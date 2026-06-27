#!/usr/bin/env bash
# Host-RAM-SAFE leftprimary cache rebuild on 4090d. The 8-concurrent version OOM'd the
# SHARED box (PyAV codec alloc -> av.error.MemoryError [Errno 12]; processes died 8->6->4).
# This runs all 8 jobs (4 FFS + 4 Utonia) through a concurrency-capped queue (MAXJ at a time),
# --resume so each picks up its partial cache. Lower per-proc threads. Detached tmux.
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
exec > playground/Checkpoints/precompute_lp_master.log 2>&1
MAXJ="${MAXJ:-3}"
NPW="${NPROC_PER_WORKER:-8}"
GPUS=(0 1 4 5)
SUITE_LIST=(libero_object_no_noops_1.0.0_lerobot libero_goal_no_noops_1.0.0_lerobot \
            libero_spatial_no_noops_1.0.0_lerobot libero_10_no_noops_1.0.0_lerobot)

# Job queue: FFS suites first (fast), then Utonia suites (slow long-pole). "TYPE|SUITE|GPU".
queue=()
for i in 0 1 2 3; do queue+=("ffs|${SUITE_LIST[$i]}|${GPUS[$i]}"); done
for i in 0 1 2 3; do queue+=("utonia|${SUITE_LIST[$i]}|${GPUS[$i]}"); done

run_job() {
  local type="$1" suite="$2" gpu="$3"
  if [ "$type" = ffs ]; then
    CUDA_VISIBLE_DEVICES="$gpu" NPROC_PER_WORKER="$NPW" SUITES="$suite" \
      LOGNAME="ffs_precompute_${suite}" bash scripts/4090d/run_precompute_ffs_4090d.sh
  else
    CUDA_VISIBLE_DEVICES="$gpu" NPROC_PER_WORKER="$NPW" \
      LOGNAME="utonia_precompute_${suite}" bash scripts/4090d/run_precompute_utonia_suite_4090d.sh "$suite"
  fi
}

echo "===== START queued leftprimary precompute (MAXJ=$MAXJ, NPW=$NPW) $(date -u +%FT%TZ) ====="
for entry in "${queue[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 10; done
  IFS='|' read -r t s g <<< "$entry"
  echo "[queue] launch $t $s gpu=$g $(date -u +%FT%TZ)  (running=$(jobs -rp | wc -l))"
  run_job "$t" "$s" "$g" &
  sleep 5   # stagger startup so model-load RAM spikes don't coincide
done
wait
echo "===== ALL DONE $(date -u +%FT%TZ) ====="
echo "ALL_PRECOMPUTE_LEFTPRIMARY_DONE"
