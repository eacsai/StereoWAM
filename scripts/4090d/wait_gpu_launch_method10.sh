#!/usr/bin/env bash
# Runs ON 4090d. GPU-wait queue for Method #10 Utonia point-cloud runs.
set -uo pipefail

H100B_IP=10.112.2.93
H100A_IP=10.112.2.128
SSHOPT="-o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=15"
REPO=/mnt/data/wangqiwei/wangqiwei/starVLA
CK=$REPO/playground/Checkpoints
LOG=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/wait_method10.log
GPU_FREE_MB=4000
POLL=600
SETTLE=300
LAUNCH_WATCH_SECONDS=1500
LAUNCH_WATCH_POLL=30

RIDS=(
  qwen3p5_0p8b_utonia_perpatch_fromscratch_30k
  qwen3p5_0p8b_utonia_resampler_fromscratch_30k
)
PORTS=( 29762 29763 )
KINDS=( utonia_perpatch utonia_resampler )
CLAIMED_GPUS=()

log(){ echo "[m10-wait $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
hssh(){ ssh $SSHOPT "root@$1" "$2" 2>/dev/null; }

gpu_claimed(){
  local key="$1:$2" seen
  for seen in "${CLAIMED_GPUS[@]}"; do
    [ "$seen" = "$key" ] && return 0
  done
  return 1
}

claim_gpu(){
  local key="$1:$2"
  gpu_claimed "$1" "$2" || CLAIMED_GPUS+=("$key")
}

release_gpu(){
  local key="$1:$2" seen
  local kept=()
  for seen in "${CLAIMED_GPUS[@]}"; do
    [ "$seen" = "$key" ] || kept+=("$seen")
  done
  CLAIMED_GPUS=("${kept[@]}")
}

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
    gpu_claimed "$ip" "$g" && continue
    pass1=$(hssh "$ip" "mb=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -dc 0-9); apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | grep -c .); [ -n \"\$mb\" ] && [ \"\$mb\" -lt $GPU_FREE_MB ] && [ \"\$apps\" -eq 0 ] && echo OK")
    [ "$pass1" = OK ] || continue
    sleep 20
    pass2=$(hssh "$ip" "mb=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -dc 0-9); apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | grep -c .); [ -n \"\$mb\" ] && [ \"\$mb\" -lt $GPU_FREE_MB ] && [ \"\$apps\" -eq 0 ] && echo OK")
    [ "$pass2" = OK ] && { echo "$g"; return 0; }
  done
  return 1
}

build_cmd(){
  local kind="$1" rid="$2" port="$3" gpu="$4" common base launcher framework train_pat
  common="cd $REPO && RUN_ID=$rid GPUS=$gpu PORT=$port"
  base="NUM_PROCESSES=1 MAX_STEPS=30000 SAVE_INTERVAL=10000 BS=32 \
DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml \
DATA_ROOT=playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW DATA_MIX=libero_all_sfstereo_rightprimary \
CAM_BRANCH=0 CAM_ROPE=0 PRETRAINED_CKPT= FREEZE_MODULES= TRAIN_ONLY="
  case "$kind" in
    utonia_perpatch)
      framework="QwenGR00T_UtoniaPerPatchAddFFS"
      launcher="scripts/h100b/run_qwen0p8_groot_utonia_perpatch.sh" ;;
    utonia_resampler)
      framework="QwenGR00T_UtoniaResamplerFFS"
      launcher="scripts/h100b/run_qwen0p8_groot_utonia_resampler.sh" ;;
    *) return 1 ;;
  esac
  train_pat="[${rid:0:1}]${rid:1}"
  echo "$common $base FRAMEWORK=$framework bash -lc 'ck_seen_before=0; if [ -d $CK/${rid}/checkpoints ] || ls $CK/${rid}/checkpoints/steps_*_pytorch_model.pt >/dev/null 2>&1; then ck_seen_before=1; fi; setsid nohup bash $launcher >> $CK/${rid}.log 2>&1 < /dev/null & launcher_pid=\$!; deadline=\$(( \$(date +%s) + $LAUNCH_WATCH_SECONDS )); while [ \"\$(date +%s)\" -lt \"\$deadline\" ]; do if pgrep -f \"run_id $train_pat\" >/dev/null 2>&1; then echo LAUNCH_ALIVE; exit 0; fi; if [ \"\$ck_seen_before\" -eq 0 ] && { [ -d $CK/${rid}/checkpoints ] || ls $CK/${rid}/checkpoints/steps_*_pytorch_model.pt >/dev/null 2>&1; }; then echo LAUNCH_ALIVE; exit 0; fi; if ! kill -0 \"\$launcher_pid\" >/dev/null 2>&1; then echo LAUNCH_FAILED; exit 1; fi; sleep $LAUNCH_WATCH_POLL; done; echo LAUNCH_FAILED_TIMEOUT; exit 1'"
}

launch_job(){
  local ip="$1" kind="$2" rid="$3" port="$4" gpu="$5" cmd
  cmd=$(build_cmd "$kind" "$rid" "$port" "$gpu") || { log "  unknown kind $kind"; return 1; }
  log "launching [$kind] $rid on host $ip GPU $gpu (PORT $port)"
  hssh "$ip" "$cmd"
}

log "=== method10 Utonia GPU-wait launcher started (order: ${KINDS[*]}) ==="
for i in "${!RIDS[@]}"; do
  rid="${RIDS[$i]}"; port="${PORTS[$i]}"; kind="${KINDS[$i]}"
  if arm_handled "$rid"; then log "[$kind] $rid already live/checkpointed -- skipping"; continue; fi
  log "--- waiting for a free GPU to launch [$kind] $rid ---"
  launched=0
  launch_failures=0
  while [ "$launched" = 0 ]; do
    if arm_handled "$rid"; then log "[$kind] $rid became handled while waiting -- next"; launched=1; break; fi
    for spec in "$H100B_IP 0 1" "$H100A_IP 0"; do
      set -- $spec; ip="$1"; shift; cands="$*"
      g=$(free_gpu_on "$ip" "$cands") || continue
      claim_gpu "$ip" "$g"
      res=$(launch_job "$ip" "$kind" "$rid" "$port" "$g")
      if echo "$res" | grep -q LAUNCH_ALIVE; then
        log "  OK [$kind] $rid training/checkpoint observed on $ip GPU $g"
        launched=1
        break
      fi
      release_gpu "$ip" "$g"
      launch_failures=$((launch_failures + 1))
      log "  ERROR [$kind] $rid launch on $ip GPU $g failed before training/checkpoint appeared -- see $CK/${rid}.log"
      log "  launcher result: $(printf '%s' "$res" | tr '\n' ' ')"
      if [ "$launch_failures" -gt 1 ]; then
        log "  HARD FAIL [$kind] $rid after retry; claimed GPU released, watcher exiting"
        exit 1
      fi
      log "  released $ip GPU $g; retrying this arm once"
    done
    [ "$launched" = 0 ] && sleep "$POLL"
  done
  log "  settling ${SETTLE}s before next job"; sleep "$SETTLE"
done
log "=== method10 jobs launched/handled -- watcher exiting ==="
