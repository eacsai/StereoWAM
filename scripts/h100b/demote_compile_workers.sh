#!/usr/bin/env bash
# Demote torch inductor compile workers to nice +19 so the co-located trainings'
# host threads get CPU first (observed: 132 workers drove load to ~100 and both
# H100s starved at ~110W waiting for kernel launches). renice only — never kills
# anything, never touches training processes. Re-scans every 2 min to catch
# newly spawned workers; exits when no train_starvla remains, or after 8h.
set -uo pipefail
CKPT=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints
LOG=$CKPT/compile_worker_renice.log
MAX_SECONDS=$((8 * 3600))
start=$(date +%s)
ts(){ date -u +%FT%TZ; }
log(){ echo "[renice $(ts)] $*" >> "$LOG"; }

log "=== started: demoting inductor compile workers to nice +19 ==="
while true; do
  pids=$(ps -eo pid,ni,args | grep "_inductor/compile_worke[r]" | awk '$2 < 19 {print $1}')
  if [ -n "$pids" ]; then
    n=$(printf '%s\n' "$pids" | wc -l)
    # shellcheck disable=SC2086  # word-splitting of the pid list is intended
    renice 19 $pids > /dev/null 2>&1
    log "demoted $n workers"
  fi
  pgrep -f "train_starvl[a]" > /dev/null || { log "no training processes left; exiting"; exit 0; }
  now=$(date +%s)
  [ $((now - start)) -gt "$MAX_SECONDS" ] && { log "safety timeout reached; exiting"; exit 0; }
  sleep 120
done
