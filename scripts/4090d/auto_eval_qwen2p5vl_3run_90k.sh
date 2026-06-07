#!/usr/bin/env bash
# Auto-eval daemon: 3 runs x every-10k (10k..90k), 4 suites, per-config video_keys, dynamic free-GPU.
# stereo = RIGHT-first (right_view,primary) per user 2026-06-04. Deletes local ckpt after eval (save disk).
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
EVAL=scripts/4090d/eval_qwen2p5vl_4suite.sh
RESULTS=playground/Checkpoints/qwen2p5vl_3run_90k_eval_results.txt
LOG=playground/Checkpoints/auto_eval_qwen2p5vl_3run_90k.log
H100B="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=90 -o ServerAliveInterval=15 root@10.112.2.93"
H100B_CKPT=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints
FREE_MB=12000
log(){ echo "[autoeval90k $(date -u +%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }

# (run_id, video_keys) — stereo RIGHT-first
RUNS=(
  "qwen2p5vl3b_4suite_stereo_primaryright_90k|right_view,primary"
  "qwen2p5vl3b_4suite_monoprimary_90k|primary"
  "qwen2p5vl3b_4suite_primarywrist_90k|primary,wrist"
)
STEPS="10000 20000 30000 40000 50000 60000 70000 80000 90000"

pick_free_gpus(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' -v t="$FREE_MB" '$2<t{print $2" "$1}' | sort -n | head -2 | awk '{print $2}' | tr '\n' ' '; }

eval_one(){
  local rid="$1" vk="$2" step="$3"
  local remote_ck="${H100B_CKPT}/${rid}/checkpoints/steps_${step}_pytorch_model.pt"
  while true; do
    local sz; sz=$($H100B "stat -c %s ${remote_ck}" 2>/dev/null | tr -dc 0-9)
    [ -n "${sz:-}" ] && [ "${sz:-0}" -gt 1000000 ] && break
    sleep 120
  done
  log "$rid step$step ckpt ready on h100b"
  local gpus=""
  while true; do gpus=$(pick_free_gpus); [ "$(echo $gpus|wc -w)" -ge 2 ] && break; log "wait 4 free GPUs (got '$gpus')"; sleep 120; done
  log "$rid step$step -> GPUs [$gpus] vk=$vk : eval"
  bash "$EVAL" "$rid" "$step" "$vk" "$gpus" >> "$LOG" 2>&1 || { log "WARN eval nonzero $rid step$step -- NOT recorded (fail-closed)"; return; }
  local edir="playground/Checkpoints/${rid}/eval_step${step}_$(echo "$vk"|tr ',' '_')"
  { echo "===== $rid step=$step vk=$vk ($(date -u)) ====="
    for s in libero_spatial libero_object libero_goal libero_10; do
      sr=$(grep 'Total success rate' "${edir}/${s}/client.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' | tail -1)
      echo "  ${s}: SR=${sr:-NA}"; done; } | tee -a "$RESULTS"
  log "EVAL_DONE ${rid} step${step}"
  # free disk: drop local ckpt after eval (re-pullable from h100b if ever needed)
  rm -f "playground/Checkpoints/${rid}/checkpoints/steps_${step}_pytorch_model.pt" 2>/dev/null || true
}

log "=== 3-run 90k eval daemon start (3 runs x 9 steps; stereo=right_view,primary) ==="
for row in "${RUNS[@]}"; do
  IFS='|' read -r rid vk <<< "$row"
  for step in $STEPS; do eval_one "$rid" "$vk" "$step"; done
done
log "=== ALL 90k EVAL DONE ==="
