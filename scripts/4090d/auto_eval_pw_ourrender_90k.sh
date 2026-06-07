#!/usr/bin/env bash
# Auto-eval daemon (90k, every 10k). 2026-06-05 hardened:
#  - flock mutex (LOCK shared by BOTH diagnostic daemons): only ONE run evals on
#    4090d at a time -> no cross-daemon GPU collision.
#  - eval script runs 1 policy-server / GPU, batched -> no 2-per-GPU over-subscription.
#  - record to RESULTS only when ALL 4 suites returned a numeric SR; retry up to
#    MAX_ATTEMPTS; if still incomplete, leave the step UNrecorded so a later daemon
#    run retries it (and keep the local ckpt to avoid re-pulling 8GB).
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
EVAL=scripts/4090d/eval_qwen2p5vl_4suite.sh
RID=qwen2p5vl3b_4suite_primarywrist_ourrender_90k
VK="primary,wrist"
RESULTS=playground/Checkpoints/qwen2p5vl_pw_ourrender_90k_eval_results.txt
LOG=playground/Checkpoints/auto_eval_pw_ourrender_90k.log
LOCK=playground/Checkpoints/.eval_4090d.lock     # shared by BOTH diagnostic daemons
H100B="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=90 -o ServerAliveInterval=15 root@10.112.2.93"
H100B_CKPT=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints
NEED_MB=15000   # per-GPU FREE memory required to host one eval lane (server+sim ~13GB + margin)
MAXGPU=2        # use AT MOST this many cards (politeness cap); proceeds with >=1
MAX_ATTEMPTS=3
STEPS="10000 20000 30000 40000 50000 60000 70000 80000 90000"
log(){ echo "[pweval $(date -u +%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }
# Dynamic: pick GPUs with >= NEED_MB FREE memory, preferring lowest utilization
# (idle first), up to MAXGPU of them. Returns space-separated indices (may be empty).
pick_eval_gpus(){ nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' -v need="$NEED_MB" '($2+0)>=need{print ($3+0)" "($2+0)" "$1}' \
  | sort -k1,1n -k2,2nr | head -$MAXGPU | awk '{print $3}' | tr '\n' ' '; }

touch "$LOCK" 2>/dev/null || true
log "=== eval daemon start RID=$RID (vk=$VK, flock + 1-server-per-GPU + retry) ==="
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
    # Acquire the shared 4090d eval lock FIRST, THEN pick GPUs reflecting the CURRENT
    # occupancy, THEN eval — so the two daemons never pick the same card concurrently
    # and the choice isn't stale by the time we run.
    exec 9>"$LOCK"
    if ! flock -w 5400 9; then log "step$step attempt$attempt — lock wait timeout, retry"; exec 9>&-; sleep 60; continue; fi
    gpus=""
    while true; do
      gpus=$(pick_eval_gpus)
      [ -n "$(echo $gpus | tr -d ' ')" ] && break
      log "no GPU with >=${NEED_MB}MB free on 4090d — wait 120s (holding eval lock)"; sleep 120
    done
    log "$RID step$step attempt$attempt -> dynamic GPUs [$gpus] vk=$VK : eval (lock held)"
    bash "$EVAL" "$RID" "$step" "$VK" "$gpus" >> "$LOG" 2>&1 || log "WARN eval nonzero step$step attempt$attempt"
    flock -u 9; exec 9>&-   # release the 4090d eval lock
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
