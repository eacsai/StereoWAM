#!/usr/bin/env bash
# Parameterized auto-eval daemon for ONE single-frame Qwen3.5-0.8B 30k run (primary+wrist).
# Faithful clone of the battle-tested auto_eval_pw_official_90k.sh:
#   flock mutex (shared .eval_4090d.lock -> only ONE run evals on 4090d at a time)
#   + dynamic GPU pick (>=NEED_MB free, idle-first, up to MAXGPU)
#   + 1 policy-server / GPU (no over-subscription)
#   + record to RESULTS only when ALL 4 suites returned a numeric SR; retry MAX_ATTEMPTS
#   + delete the LOCAL pulled ckpt copy after recording (h100b keeps the real one).
# Steps 10k/20k/30k. Launch 3x (one per run); they serialize via the shared lock.
# Usage: auto_eval_0p8b_30k_run.sh <RUN_ID>
# NOTE: multi-frame (3-frame/4D) runs ARE supported — eval_libero.py has a generic
#       observation_indices rolling-buffer client; obs_indices/video_keys/num_frames are
#       derived from the ckpt's DataConfig (single source) by eval_qwen2p5vl_4suite.sh.
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
RID="${1:?need RUN_ID}"
EVAL=scripts/4090d/eval_qwen2p5vl_4suite.sh
VK="${2:-primary,wrist}"
RESULTS=playground/Checkpoints/${RID}_eval_results.txt
LOG=playground/Checkpoints/auto_eval_${RID}.log
LOCK=playground/Checkpoints/.eval_4090d.lock          # shared by ALL eval daemons
H100B="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=90 -o ServerAliveInterval=15 root@10.112.2.93"
H100B_CKPT=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints
NEED_MB=15000   # per-GPU FREE memory to host one eval lane (server+sim ~13GB + margin)
MAXGPU=2        # politeness cap; proceeds with >=1
MAX_ATTEMPTS=3
STEPS="10000 20000 30000"
log(){ echo "[eval $(date -u +%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }
pick_eval_gpus(){ nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' -v need="$NEED_MB" '($2+0)>=need{print ($3+0)" "($2+0)" "$1}' \
  | sort -k1,1n -k2,2nr | head -$MAXGPU | awk '{print $3}' | tr '\n' ' '; }

touch "$LOCK" 2>/dev/null || true
log "=== eval daemon start RID=$RID (vk=$VK, flock + 1-server/GPU + retry) ==="
for step in $STEPS; do
  if grep -q "step=${step} " "$RESULTS" 2>/dev/null; then log "step$step already recorded — skip"; continue; fi
  remote_ck="${H100B_CKPT}/${RID}/checkpoints/steps_${step}_pytorch_model.pt"
  while true; do
    sz=$($H100B "stat -c %s ${remote_ck}" 2>/dev/null | tr -dc 0-9)
    [ -n "${sz:-}" ] && [ "${sz:-0}" -gt 1000000 ] && break
    sleep 120
  done
  log "$RID step$step ckpt ready"
  edir="playground/Checkpoints/${RID}/eval_step${step}_$(echo "$VK"|tr ',' '_')"
  recorded=0
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    exec 9>"$LOCK"
    if ! flock -w 5400 9; then log "step$step a$attempt lock wait timeout, retry"; exec 9>&-; sleep 60; continue; fi
    gpus=""
    while true; do
      gpus=$(pick_eval_gpus)
      [ -n "$(echo $gpus | tr -d ' ')" ] && break
      log "no GPU with >=${NEED_MB}MB free on 4090d — wait 120s (holding lock)"; sleep 120
    done
    log "$RID step$step attempt$attempt -> dynamic GPUs [$gpus] vk=$VK : eval (lock held)"
    bash "$EVAL" "$RID" "$step" "$VK" "$gpus" >> "$LOG" 2>&1 || log "WARN eval nonzero step$step attempt$attempt"
    flock -u 9; exec 9>&-
    ok=0; block="===== $RID step=${step} vk=$VK ($(date -u)) ====="$'\n'
    for s in libero_spatial libero_object libero_goal libero_10; do
      sr=$(grep 'Total success rate' "${edir}/${s}/client.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' | tail -1)
      [ -n "$sr" ] && ok=$((ok+1))
      block+="  ${s}: SR=${sr:-FAILED}"$'\n'
    done
    printf '%s' "$block" | tee -a "$LOG"
    if [ "$ok" -ge 4 ]; then
      printf '%s' "$block" >> "$RESULTS"
      log "step$step RECORDED (4/4 ok, attempt$attempt)"
      recorded=1; break
    else
      log "step$step INCOMPLETE ($ok/4) attempt$attempt — retry"; sleep 60
    fi
  done
  if [ "$recorded" = 1 ]; then
    rm -f "playground/Checkpoints/${RID}/checkpoints/steps_${step}_pytorch_model.pt" 2>/dev/null || true
  else
    log "step$step STILL INCOMPLETE after $MAX_ATTEMPTS attempts — left UNrecorded (restart daemon to retry); keeping local ckpt"
  fi
done
log "=== ALL EVAL DONE RID=$RID ==="
