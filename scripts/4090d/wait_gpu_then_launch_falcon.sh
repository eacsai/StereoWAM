#!/usr/bin/env bash
# Runs ON 4090d. Waits for a free GPU on h100b/h100a and launches the FFS #9
# (FALCON-style spatial injector, FFS source) from-scratch 30k arm, then exits.
# Safety: only grabs a GPU double-confirmed idle (mem<4GB AND no compute apps);
# skips if the arm is already live/checkpointed; report-only crash handling.
set -uo pipefail

H100B_IP=10.112.2.93
H100A_IP=10.112.2.128
SSHOPT="-o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=15"
REPO=/mnt/data/wangqiwei/wangqiwei/starVLA
CK=$REPO/playground/Checkpoints
LAUNCHER=scripts/h100b/run_qwen0p8_groot_falcon_spatial_inject.sh
LOG=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/wait_falcon_launcher.log
GPU_FREE_MB=4000
POLL=600
SETTLE=300
ARM_RID=qwen3p5_0p8b_ffs_falcon_spatial_inject_fromscratch_30k
ARM_PORT=29744

log(){ echo "[falcon-wait $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
hssh(){ ssh $SSHOPT "root@$1" "$2" 2>/dev/null; }

# arm already live OR already has a checkpoint -> handled. bracket first char so the
# remote pgrep does not self-match its own bash -c wrapper.
arm_handled(){
  local rid="$1" h
  local bp="[${rid:0:1}]${rid:1}"
  for h in "$H100B_IP" "$H100A_IP"; do
    if hssh "$h" "pgrep -f 'run_id ${bp}' >/dev/null 2>&1"; then return 0; fi
    if hssh "$h" "ls ${CK}/${rid}/checkpoints/steps_*_pytorch_model.pt >/dev/null 2>&1"; then return 0; fi
  done
  return 1
}

free_gpu_on(){
  local ip="$1" cands="$2" g pass1 pass2
  for g in $cands; do
    pass1=$(hssh "$ip" "mb=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -dc 0-9); apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | grep -c .); [ -n \"\$mb\" ] && [ \"\$mb\" -lt $GPU_FREE_MB ] && [ \"\$apps\" -eq 0 ] && echo OK")
    [ "$pass1" = OK ] || continue
    sleep 20
    pass2=$(hssh "$ip" "mb=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -dc 0-9); apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | grep -c .); [ -n \"\$mb\" ] && [ \"\$mb\" -lt $GPU_FREE_MB ] && [ \"\$apps\" -eq 0 ] && echo OK")
    [ "$pass2" = OK ] && { echo "$g"; return 0; }
  done
  return 1
}

launch_arm(){
  local ip="$1" rid="$2" port="$3" gpu="$4"
  log "launching $rid on host $ip GPU $gpu (PORT $port)"
  local bp="[${rid:0:1}]${rid:1}"
  hssh "$ip" "cd $REPO && RUN_ID=$rid GPUS=$gpu PORT=$port MAX_STEPS=30000 setsid nohup bash $LAUNCHER >> $CK/${rid}.log 2>&1 < /dev/null & sleep 18; pgrep -f 'run_id ${bp}' >/dev/null 2>&1 && echo LAUNCH_ALIVE || echo LAUNCH_DEAD"
}

log "=== FFS#9 FALCON GPU-wait launcher started (single arm: falcon_spatial_inject fromscratch) ==="
while true; do
  if arm_handled "$ARM_RID"; then
    log "=== FFS#9 arm launched/handled — watcher exiting ==="
    exit 0
  fi
  launched=0
  for spec in "$H100B_IP 0 1" "$H100A_IP 0"; do
    set -- $spec; ip="$1"; shift; cands="$*"
    g=$(free_gpu_on "$ip" "$cands") || continue
    res=$(launch_arm "$ip" "$ARM_RID" "$ARM_PORT" "$g")
    if echo "$res" | grep -q LAUNCH_ALIVE; then
      log "  $ARM_RID is alive on $ip GPU $g"; launched=1
    else
      log "  ⚠️ $ARM_RID launch on $ip GPU $g did not survive 18s (guard/preflight refusal?) — see $CK/${ARM_RID}.log"
    fi
    break
  done
  if [ "$launched" = 1 ]; then
    log "  settling ${SETTLE}s before next sweep"; sleep "$SETTLE"
  else
    sleep "$POLL"
  fi
done
