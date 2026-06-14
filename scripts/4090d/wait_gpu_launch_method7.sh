#!/usr/bin/env bash
# Runs ON 4090d. GPU-wait queue for #7 ControlVLA K/V branch, 2 arms:
#   fromscratch (ablation)  then  warmstartB (faithful ControlVLA: frozen trunk + zero-init control branch).
# Mirrors wait_gpu_then_launch_baselines_then_falcon.sh exactly; only the job table + build_cmd differ.
# One job per double-confirmed-idle GPU (mem<4GB AND no compute apps). Idempotent (skips live/checkpointed).
set -uo pipefail

H100B_IP=10.112.2.93
H100A_IP=10.112.2.128
SSHOPT="-o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=15"
REPO=/mnt/data/wangqiwei/wangqiwei/starVLA
CK=$REPO/playground/Checkpoints
LOG=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/wait_method7.log
B_CKPT=playground/Checkpoints/qwen3p5_0p8b_4suite_stereo_camrope_rightprimary_ourrender_30k/checkpoints/steps_30000_pytorch_model.pt
TRAIN_ONLY_ALLOW="ffs_pos_emb,.attn1.to_k_z.,.attn1.to_v_z."
GPU_FREE_MB=4000
POLL=600
SETTLE=300

RIDS=(
  qwen3p5_0p8b_ffs_controlvla_branch_fromscratch_30k
  qwen3p5_0p8b_ffs_controlvla_branch_warmstartB_30k
)
PORTS=( 29750 29751 )
KINDS=( cv_fromscratch cv_warmstartB )

log(){ echo "[m7-wait $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
hssh(){ ssh $SSHOPT "root@$1" "$2" 2>/dev/null; }

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

build_cmd(){
  local kind="$1" rid="$2" port="$3" gpu="$4" common base envv launcher
  common="cd $REPO && RUN_ID=$rid GPUS=$gpu PORT=$port"
  launcher="scripts/h100b/run_qwen0p8_groot_controlvla.sh"
  base="NUM_PROCESSES=1 MAX_STEPS=30000 SAVE_INTERVAL=10000 BS=32 \
DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml FRAMEWORK=QwenGR00T_ControlVLAFFS \
DATA_ROOT=playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW DATA_MIX=libero_all_sfstereo_rightprimary \
CAM_BRANCH=0 CAM_ROPE=0"
  case "$kind" in
    cv_fromscratch)
      envv="$base PRETRAINED_CKPT= FREEZE_MODULES= TRAIN_ONLY=" ;;
    cv_warmstartB)
      envv="$base PRETRAINED_CKPT=$B_CKPT FREEZE_MODULES= TRAIN_ONLY=$TRAIN_ONLY_ALLOW" ;;
    *) return 1 ;;
  esac
  echo "$common $envv setsid nohup bash $launcher >> $CK/${rid}.log 2>&1 < /dev/null & sleep 20; pgrep -f 'run_id [${rid:0:1}]${rid:1}' >/dev/null 2>&1 && echo LAUNCH_ALIVE || echo LAUNCH_DEAD"
}

launch_job(){
  local ip="$1" kind="$2" rid="$3" port="$4" gpu="$5" cmd
  cmd=$(build_cmd "$kind" "$rid" "$port" "$gpu") || { log "  unknown kind $kind"; return 1; }
  log "launching [$kind] $rid on host $ip GPU $gpu (PORT $port)"
  hssh "$ip" "$cmd"
}

log "=== method7 ControlVLA GPU-wait launcher started (order: ${KINDS[*]}) ==="
for i in "${!RIDS[@]}"; do
  rid="${RIDS[$i]}"; port="${PORTS[$i]}"; kind="${KINDS[$i]}"
  if arm_handled "$rid"; then log "[$kind] $rid already live/checkpointed — skipping"; continue; fi
  log "--- waiting for a free GPU to launch [$kind] $rid ---"
  launched=0
  while [ "$launched" = 0 ]; do
    if arm_handled "$rid"; then log "[$kind] $rid became handled while waiting — next"; launched=1; break; fi
    for spec in "$H100B_IP 0 1" "$H100A_IP 0"; do
      set -- $spec; ip="$1"; shift; cands="$*"
      g=$(free_gpu_on "$ip" "$cands") || continue
      res=$(launch_job "$ip" "$kind" "$rid" "$port" "$g")
      if echo "$res" | grep -q LAUNCH_ALIVE; then log "  OK [$kind] $rid alive on $ip GPU $g"; launched=1
      else log "  WARN [$kind] $rid launch on $ip GPU $g did not survive 20s — see $CK/${rid}.log"; launched=1; fi
      break
    done
    [ "$launched" = 0 ] && sleep "$POLL"
  done
  log "  settling ${SETTLE}s before next job"; sleep "$SETTLE"
done
log "=== method7 jobs launched/handled — watcher exiting ==="
