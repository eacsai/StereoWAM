#!/usr/bin/env bash
# One-shot watcher: launch the cam_branch mechanism-gate run (warm-start B, frozen
# VLM, 2000 steps) once an h100b GPU is GENUINELY free because a #4 arm FINISHED.
# Hardened (Fable-5 review HIGH): a crash-freed GPU must NOT fire the gate —
#   (1) completion signal: at least one h100b #4 arm has its steps_30000 checkpoint;
#   (2) the candidate GPU has ZERO compute apps (not just low memory — a process in
#       its model-loading window can sit under 4GB for minutes);
#   (3) double-confirmation: the same GPU must pass both checks again after 150s;
#   (4) anchored scheduler pgrep + hourly heartbeat + crash-sentinel logging.
# Never kills anything.
set -uo pipefail
REPO=/mnt/data/wangqiwei/wangqiwei/starVLA
CKPT=$REPO/playground/Checkpoints
Q=$CKPT/ffs4_ablation_queue
RID=qwen3p5_0p8b_stereo_cam_branch_prope_warmstartB_mechgate_2k
WLOG=$CKPT/cam_branch_mechgate_watcher.log
GPU_FREE_MB=4000
KEEP_CKPT=$CKPT/qwen3p5_0p8b_ffs_depthtoken_keep_fromscratch_30k/checkpoints/steps_30000_pytorch_model.pt
STRIP_CKPT=$CKPT/qwen3p5_0p8b_ffs_depthtoken_strip_fromscratch_30k/checkpoints/steps_30000_pytorch_model.pt
ts(){ date -u +%FT%TZ; }
log(){ echo "[mechgate-watch $(ts)] $*" >> "$WLOG"; }

gpu_idle(){  # gpu -> 0 if low-mem AND no compute apps
  local g="$1" mb apps
  mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -dc 0-9)
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | grep -c . || true)
  [ -n "$mb" ] && [ "$mb" -lt "$GPU_FREE_MB" ] && [ "$apps" -eq 0 ]
}

log "=== hardened watcher started: need 3 claims + scheduler exit + a #4 30k ckpt + double-confirmed idle GPU ==="
loop=0
while true; do
  loop=$((loop + 1))
  n_claims=$(ls -d "$Q"/claim_qwen3p5_0p8b_ffs_* 2>/dev/null | wc -l)
  sched_alive=$(pgrep -fc "bash .*ffs4_ablation_scheduler\.sh" || true)
  done_ckpt=""
  [ -f "$KEEP_CKPT" ] && done_ckpt=keep
  [ -z "$done_ckpt" ] && [ -f "$STRIP_CKPT" ] && done_ckpt=strip

  if [ $((loop % 30)) -eq 0 ]; then
    log "heartbeat: claims=$n_claims sched_alive=$sched_alive done_ckpt=${done_ckpt:-none}"
  fi

  if [ "$n_claims" -ge 3 ] && [ "${sched_alive:-0}" -eq 0 ]; then
    for g in 0 1; do
      if gpu_idle "$g"; then
        if [ -z "$done_ckpt" ]; then
          log "WARNING: GPU $g is idle but NO #4 30k checkpoint exists — possible training crash; NOT launching. Human attention needed."
          break
        fi
        log "GPU $g idle and $done_ckpt finished (30k ckpt present); double-confirming after 150s"
        sleep 150
        if gpu_idle "$g"; then
          log "GPU $g confirmed idle; launching mechgate"
          cd "$REPO"
          GPUS="$g" setsid nohup bash scripts/h100b/run_qwen0p8_groot_cam_branch.sh \
            >> "$CKPT/${RID}.log" 2>&1 < /dev/null &
          log "mechgate pid $! on GPU $g; watcher exiting"
          exit 0
        fi
        log "GPU $g no longer idle on re-check; continuing watch"
      fi
    done
  fi
  sleep 120
done
