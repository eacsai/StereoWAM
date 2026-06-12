#!/usr/bin/env bash
# Runs ON 4090d. Overnight-unattended: waits for free GPUs on h100b/h100a and
# launches the two FFS #5 (LLaMA-Adapter prefix) arms — one per freed GPU — then exits.
#   arm 1: qwen3p5_0p8b_ffs_llama_adapter_prefix_warmstartB_frozen_30k  (warm-start B, frozen VLM)
#   arm 2: qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_30k        (from scratch, full finetune)
# Safety: only grabs a GPU that is double-confirmed idle (mem<4GB AND no compute apps);
# skips an arm already live/checkpointed; the launcher's own guards are the backstop.
# Crash handling is report-only (this watcher never kills/relaunches a running job).
set -uo pipefail

H100B_IP=10.112.2.93
H100A_IP=10.112.2.128
SSHOPT="-o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=15"
REPO=/mnt/data/wangqiwei/wangqiwei/starVLA
CK=$REPO/playground/Checkpoints
LAUNCHER=scripts/h100b/run_qwen0p8_groot_llama_adapter_prefix.sh
LOG=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/wait_ffs5_launcher.log
GPU_FREE_MB=4000
POLL=600          # seconds between sweeps
SETTLE=300        # seconds to let a freshly-launched arm grab its GPU before scanning again

ARM1_RID=qwen3p5_0p8b_ffs_llama_adapter_prefix_warmstartB_frozen_30k
ARM1_PORT=29742
ARM2_RID=qwen3p5_0p8b_ffs_llama_adapter_prefix_fromscratch_perhead_30k
ARM2_PORT=29743

log(){ echo "[ffs5-wait $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
hssh(){ ssh $SSHOPT "root@$1" "$2" 2>/dev/null; }   # $1=ip $2=remote cmd

# arm already live (training process) OR already has a checkpoint -> consider it handled.
# ⚠️ bracket the run_id's first char ([q]wen...) so the remote pgrep does NOT match its own
# bash -c wrapper (whose cmdline literally contains the search pattern) — classic self-match.
arm_handled(){ # $1=run_id
  local rid="$1" h
  local bp="[${rid:0:1}]${rid:1}"
  for h in "$H100B_IP" "$H100A_IP"; do
    if hssh "$h" "pgrep -f 'run_id ${bp}' >/dev/null 2>&1"; then return 0; fi
    if hssh "$h" "ls ${CK}/${rid}/checkpoints/steps_*_pytorch_model.pt >/dev/null 2>&1"; then return 0; fi
  done
  return 1
}

# double-confirmed idle GPU on a host. echoes the gpu index, or nothing.
# idle = memory.used < GPU_FREE_MB AND zero compute-apps, on two reads 20s apart.
free_gpu_on(){ # $1=ip  $2=space-separated candidate gpu indices
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

launch_arm(){ # $1=ip $2=run_id $3=port $4=gpu
  local ip="$1" rid="$2" port="$3" gpu="$4"
  log "launching $rid on host $ip GPU $gpu (PORT $port)"
  local bp="[${rid:0:1}]${rid:1}"
  local gph=0; case "$rid" in *perhead*) gph=1;; esac   # per-head gate arm sets GATE_PER_HEAD=1
  hssh "$ip" "cd $REPO && RUN_ID=$rid GATE_PER_HEAD=$gph GPUS=$gpu PORT=$port MAX_STEPS=30000 setsid nohup bash $LAUNCHER >> $CK/${rid}.log 2>&1 < /dev/null & sleep 18; pgrep -f 'run_id ${bp}' >/dev/null 2>&1 && echo LAUNCH_ALIVE || echo LAUNCH_DEAD"
}

log "=== FFS#5 GPU-wait launcher started (arms: frozen + fromscratch; one per freed GPU) ==="
while true; do
  arm_handled "$ARM1_RID" && a1=done || a1=pending
  arm_handled "$ARM2_RID" && a2=done || a2=pending
  if [ "$a1" = done ] && [ "$a2" = done ]; then
    log "=== both FFS#5 arms launched/handled — watcher exiting ==="
    exit 0
  fi
  # pick the next pending arm
  if [ "$a1" = pending ]; then NRID=$ARM1_RID; NPORT=$ARM1_PORT; else NRID=$ARM2_RID; NPORT=$ARM2_PORT; fi

  launched=0
  # try h100b GPU 0,1 then h100a GPU 0
  for spec in "$H100B_IP 0 1" "$H100A_IP 0"; do
    set -- $spec; ip="$1"; shift; cands="$*"
    g=$(free_gpu_on "$ip" "$cands") || continue
    res=$(launch_arm "$ip" "$NRID" "$NPORT" "$g")
    if echo "$res" | grep -q LAUNCH_ALIVE; then
      log "  $NRID is alive on $ip GPU $g"; launched=1
    else
      log "  ⚠️ $NRID launch on $ip GPU $g did not survive 18s (guard/preflight refusal?) — see $CK/${NRID}.log"
    fi
    break
  done

  if [ "$launched" = 1 ]; then
    log "  settling ${SETTLE}s before next sweep"; sleep "$SETTLE"
  else
    sleep "$POLL"
  fi
done
