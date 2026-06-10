#!/usr/bin/env bash
# One-shot pre-relaunch gate for the cam_rope-off (#4) relaunch: runs the three
# smokes sequentially on one GPU and writes a summary. Mirrors the scheduler's
# Phase-1 gate invocation (ffs4_ablation_scheduler.sh run_smokes) but manual, so
# we can verify BEFORE restarting the schedulers (SMOKE_PASSED already exists,
# so the schedulers themselves would skip the gate).
set -uo pipefail
REPO=/mnt/data/wangqiwei/wangqiwei/starVLA
FFS_REPO=/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo
PY=/opt/conda/envs/starvla/bin/python3.10
CKPT=$REPO/playground/Checkpoints
GPU=${GPU:-0}
SUMMARY=$CKPT/ffs4_camrope_off_smokes_summary.log
cd "$REPO"
# Refuse to co-locate smoke allocations next to a live training (memory pressure can
# OOM the training's activation spikes). FORCE=1 to override deliberately.
mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -dc 0-9)
if [ "${FORCE:-0}" != "1" ]; then
  { [ -n "$mb" ] && [ "$mb" -lt "${GPU_FREE_MB:-4000}" ]; } || {
    echo "GPU $GPU busy (${mb:-unknown}MB) — refusing to smoke next to a live training (FORCE=1 to override)"
    exit 1
  }
fi
: > "$SUMMARY"
for s in smoke_camrope_disable_equivalence smoke_depthtoken_strip smoke_depthimage; do
  slog=$CKPT/ffs4_manual_${s}.log
  echo "[$(date -u +%FT%TZ)] running $s on GPU $GPU ..." >> "$SUMMARY"
  if CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO:$FFS_REPO" "$PY" "scripts/h100b/$s.py" --device cuda > "$slog" 2>&1 \
     && grep -q SMOKE_ALL_PASS "$slog"; then
    echo "[$(date -u +%FT%TZ)]   $s PASS" >> "$SUMMARY"
  else
    echo "[$(date -u +%FT%TZ)]   $s FAIL (see $slog)" >> "$SUMMARY"
    echo "GATE_FAILED" >> "$SUMMARY"
    exit 1
  fi
done
echo "GATE_ALL_PASS" >> "$SUMMARY"
