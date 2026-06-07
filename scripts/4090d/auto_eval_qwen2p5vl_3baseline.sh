#!/usr/bin/env bash
# Autonomous eval daemon for the Qwen2.5-VL 3-baseline 4-suite ablation.
# Processes a fixed (run_id, video_keys, step) matrix in time-order. For each:
#   1. wait until that step's ckpt exists on h100b
#   2. wait until 4 GPUs on this box (4090d) are free (mem_used < FREE_MB)
#   3. run eval_qwen2p5vl_4suite.sh with those GPUs and the config's video_keys
#   4. append the 4-suite SR line to the results file + emit EVAL_DONE marker
# Robust to GPU contention by picking the 4 lowest-mem free GPUs at eval start.
# Detached; survives ssh close. NEVER pkill -f matching its own cmdline.
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
EVAL=scripts/4090d/eval_qwen2p5vl_4suite.sh
H100B="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=90 -o ServerAliveInterval=15 root@10.112.2.93"
H100B_CKPT=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints
RESULTS=playground/Checkpoints/qwen2p5vl_3baseline_eval_results.txt
LOG=playground/Checkpoints/auto_eval_qwen2p5vl_3baseline.log
FREE_MB=16000          # a GPU counts as free if mem_used < this
WAIT_MAX_MIN=1800      # give up waiting for one ckpt after 30h

log(){ echo "[autoeval $(date -u +%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }

# (run_id, video_keys, step) -- in expected time-order (stereo first, then mono, then primary+wrist)
MATRIX=(
  "qwen2p5vl3b_4suite_stereo_primaryright_0604|primary,right_view|20000"
  "qwen2p5vl3b_4suite_stereo_primaryright_0604|primary,right_view|30000"
  "qwen2p5vl3b_4suite_monoprimary_0604|primary|20000"
  "qwen2p5vl3b_4suite_monoprimary_0604|primary|30000"
  "qwen2p5vl3b_4suite_primarywrist_0604|primary,wrist|20000"
  "qwen2p5vl3b_4suite_primarywrist_0604|primary,wrist|30000"
)

pick_free_gpus(){   # echo 4 free gpu indices (lowest mem first), or nothing if <4 free
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v t="$FREE_MB" '$2 < t {print $2" "$1}' | sort -n | head -4 | awk '{print $2}' | tr '\n' ' '
}

eval_one(){
  local rid="$1" vk="$2" step="$3"
  local remote_ck="${H100B_CKPT}/${rid}/checkpoints/steps_${step}_pytorch_model.pt"
  # 1. wait for ckpt on h100b
  local waited=0
  while true; do
    local sz; sz=$($H100B "stat -c %s ${remote_ck}" 2>/dev/null | tr -dc 0-9)
    [ -n "${sz:-}" ] && [ "${sz:-0}" -gt 1000000 ] && break
    waited=$((waited+1))
    [ "$waited" -ge "$((WAIT_MAX_MIN/2))" ] && { log "GIVE UP $rid step$step (ckpt never appeared)"; return 1; }
    sleep 120
  done
  log "$rid step$step ckpt ready on h100b"
  # 2. wait for 4 free GPUs
  local gpus=""
  while true; do
    gpus=$(pick_free_gpus)
    [ "$(echo $gpus | wc -w)" -ge 4 ] && break
    log "waiting for 4 free GPUs (got: '$gpus') ..."; sleep 120
  done
  log "$rid step$step -> GPUs [$gpus] vk=$vk : starting eval"
  # 3. run eval
  bash "$EVAL" "$rid" "$step" "$vk" "$gpus" >> "$LOG" 2>&1 || { log "WARN eval nonzero $rid step$step -- NOT recorded (fail-closed)"; return; }
  # 4. record
  local edir="playground/Checkpoints/${rid}/eval_step${step}_$(echo "$vk" | tr ',' '_')"
  {
    echo "===== $rid  step=$step  video_keys=$vk  ($(date -u)) ====="
    for s in libero_spatial libero_object libero_goal libero_10; do
      sr=$(grep 'Total success rate' "${edir}/${s}/client.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' | tail -1)
      echo "  ${s}: SR=${sr:-NA}"
    done
  } | tee -a "$RESULTS"
  log "EVAL_DONE ${rid} step${step}"
}

log "=== auto-eval daemon start (matrix ${#MATRIX[@]} points) ==="
for row in "${MATRIX[@]}"; do
  IFS='|' read -r rid vk step <<< "$row"
  eval_one "$rid" "$vk" "$step"
done
log "=== ALL EVAL POINTS PROCESSED ==="
