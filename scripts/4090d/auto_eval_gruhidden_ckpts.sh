#!/usr/bin/env bash
# Auto-eval gru_hidden ckpts as they appear on h100b: wait-until-fully-written ->
# transfer h100b->4090d -> right_view eval (libero_goal, 100ep) -> log SR.
# Runs on 4090d (nohup). Source of truth: <run>/auto_eval_results.txt
set -uo pipefail
RUN=pi_qwen0p8_camrope_controlvla_gruhidden_fromscratch_h100b_0529
H=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints/$RUN
DST=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/$RUN
EVAL=/data/wangqiwei/ICLR2026/starVLA/scripts/4090d/eval_one_ckpt.sh
R="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 root@10.112.2.93"
RESULTS=$DST/auto_eval_results.txt
GPU=${GPU:-3}; PORT=${PORT:-6712}
mkdir -p "$DST/checkpoints"
echo "[auto-eval] start $(date -u +%H:%M) GPU=$GPU" >> "$RESULTS"

for STEP in 5000 10000 15000 20000 25000 30000; do
  CK=$DST/checkpoints/steps_${STEP}_pytorch_model.pt
  EVDIR=$DST/eval_logs_rightview/step${STEP}
  grep -q "step${STEP}:" "$RESULTS" 2>/dev/null && { echo "[skip] step$STEP done"; continue; }

  # wait until the h100b ckpt is fully written (size > 1MB and stable across 20s), up to ~150 min
  echo "[wait] step$STEP $(date -u +%H:%M)"
  prev=-1; ok=0
  for i in $(seq 1 450); do
    sz=$($R "stat -c %s $H/checkpoints/steps_${STEP}_pytorch_model.pt 2>/dev/null" 2>/dev/null || echo 0)
    sz=${sz:-0}
    if [ "$sz" -gt 1000000 ] && [ "$sz" = "$prev" ]; then ok=1; break; fi
    prev=$sz; sleep 20
  done
  [ "$ok" != 1 ] && { echo "step${STEP}: TIMEOUT_WAIT_CKPT ($(date -u +%H:%M))" >> "$RESULTS"; continue; }

  echo "[transfer] step$STEP $sz bytes $(date -u +%H:%M)"
  $R "cat $H/checkpoints/steps_${STEP}_pytorch_model.pt" > "$CK"
  lsz=$(stat -c %s "$CK" 2>/dev/null || echo 0)
  [ "$lsz" != "$sz" ] && { echo "step${STEP}: TRANSFER_MISMATCH h100b=$sz 4090d=$lsz" >> "$RESULTS"; continue; }

  echo "[eval] step$STEP $(date -u +%H:%M)"
  bash "$EVAL" "$CK" "$GPU" "$PORT" "$EVDIR" libero_goal primary,right_view > "$DST/auto_eval_step${STEP}.log" 2>&1
  SR=$(grep "Total success rate" "$EVDIR/client.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  echo "step${STEP}: SR=${SR:-FAILED} ($(date -u +%H:%M))" >> "$RESULTS"
  echo "[done] step$STEP SR=${SR:-FAILED}"
done
echo "[auto-eval] ALL DONE $(date -u +%H:%M)" >> "$RESULTS"
