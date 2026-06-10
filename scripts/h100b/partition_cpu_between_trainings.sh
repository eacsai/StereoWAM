#!/usr/bin/env bash
# Split CPU affinity between the two co-located h100b trainings (keep vs strip):
# both spawn ~96 OMP/compile threads sized for the whole 192-core box, and the two
# pools trample each other (observed: load 100+, GPUs starved at ~110W, steps
# 1.35s -> 10-19s). Pinning each training's WHOLE session to its own NUMA node
# (48 physical cores + HT siblings + local memory) removes the contention.
# taskset -apc is live and reversible (re-run with 0-191 to undo); never kills.
# Newly forked children (dataloader workers, compile workers) inherit the mask.
set -uo pipefail
# NUMA node0 = 0-47,96-143 ; node1 = 48-95,144-191 (lscpu, 2026-06-10)
KEEP_CPUS=${KEEP_CPUS:-0-47,96-143}
STRIP_CPUS=${STRIP_CPUS:-48-95,144-191}

pin_session(){ # run_id cpus label
  local rid="$1" cpus="$2" label="$3"
  local main sid members count=0
  main=$(pgrep -f "train_starvla.py.*run_id ${rid}" | head -1)
  if [ -z "$main" ]; then
    echo "[pin] $label: no live training process found for run_id ${rid} — skipped"
    return 1
  fi
  sid=$(ps -o sess= -p "$main" | tr -dc 0-9)
  members=$(ps -eo pid,sess | awk -v s="$sid" '$2==s {print $1}')
  for p in $members; do
    taskset -apc "$cpus" "$p" > /dev/null 2>&1 && count=$((count + 1))
  done
  echo "[pin] $label: session $sid, $count processes (all threads) pinned to CPUs $cpus"
}

pin_session "qwen3p5_0p8b_ffs_depthtoken_keep_fromscratch_30k"  "$KEEP_CPUS"  keep
pin_session "qwen3p5_0p8b_ffs_depthtoken_strip_fromscratch_30k" "$STRIP_CPUS" strip
