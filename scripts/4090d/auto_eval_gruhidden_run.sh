#!/usr/bin/env bash
# Generic auto-eval for a gru_hidden run. Env: RUN_ID (req), GPU (def 4), PORT (def 6713),
# STEPS (def "5000 10000 15000 20000 25000 30000"), WAIT_MAX_MIN (def 300, how long to wait
# for the run to START on h100b). Waits for the run -> sets up 4090d eval config -> for each
# step waits-until-ckpt-written -> transfer h100b->4090d -> right_view eval -> append SR.
set -uo pipefail
RUN=${RUN_ID:?set RUN_ID}
GPU=${GPU:-4}; PORT=${PORT:-6713}
STEPS=${STEPS:-"5000 10000 15000 20000 25000 30000"}
WAIT_MAX_MIN=${WAIT_MAX_MIN:-300}
H=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints/$RUN
DST=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/$RUN
EVAL=/data/wangqiwei/ICLR2026/starVLA/scripts/4090d/eval_one_ckpt.sh
R="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 root@10.112.2.93"
mkdir -p "$DST/checkpoints"
RESULTS=$DST/auto_eval_results.txt
echo "[auto-eval $RUN] start $(date -u +%H:%M) GPU=$GPU STEPS=[$STEPS]" >> "$RESULTS"

# 1) wait for the run to exist on h100b (config.full.yaml written at launch)
ok=0
for i in $(seq 1 $WAIT_MAX_MIN); do
  $R "test -f $H/config.full.yaml" 2>/dev/null && { ok=1; break; }
  sleep 60
done
[ "$ok" = 1 ] || { echo "[auto-eval $RUN] TIMEOUT waiting for run to start" >> "$RESULTS"; exit 1; }

# 2) set up 4090d eval run-dir config (idempotent)
if [ ! -f "$DST/config.yaml" ]; then
  $R "cat $H/config.full.yaml" > "$DST/config.yaml"
  $R "cat $H/dataset_statistics.json" > "$DST/dataset_statistics.json"
  sed -i "s#/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo#/data/wangqiwei/ICLR2026/Fast-FoundationStereo#g" "$DST/config.yaml"
  echo "[auto-eval $RUN] config set up $(date -u +%H:%M)" >> "$RESULTS"
fi

# 3) eval loop
for STEP in $STEPS; do
  CK=$DST/checkpoints/steps_${STEP}_pytorch_model.pt
  EVDIR=$DST/eval_logs_rightview/step${STEP}
  grep -q "step${STEP}:" "$RESULTS" 2>/dev/null && continue
  prev=-1; ok=0
  for i in $(seq 1 600); do   # up to ~200min per ckpt (90k steps are ~slow)
    sz=$($R "stat -c %s $H/checkpoints/steps_${STEP}_pytorch_model.pt 2>/dev/null" 2>/dev/null || echo 0); sz=${sz:-0}
    if [ "$sz" -gt 1000000 ] && [ "$sz" = "$prev" ]; then ok=1; break; fi
    prev=$sz; sleep 20
  done
  [ "$ok" != 1 ] && { echo "step${STEP}: TIMEOUT_WAIT_CKPT ($(date -u +%H:%M))" >> "$RESULTS"; continue; }
  $R "cat $H/checkpoints/steps_${STEP}_pytorch_model.pt" > "$CK"
  lsz=$(stat -c %s "$CK" 2>/dev/null || echo 0)
  [ "$lsz" != "$sz" ] && { echo "step${STEP}: TRANSFER_MISMATCH h100b=$sz 4090d=$lsz" >> "$RESULTS"; continue; }
  bash "$EVAL" "$CK" "$GPU" "$PORT" "$EVDIR" libero_goal primary,right_view > "$DST/auto_eval_step${STEP}.log" 2>&1
  SR=$(grep "Total success rate" "$EVDIR/client.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  echo "step${STEP}: SR=${SR:-FAILED} ($(date -u +%H:%M))" >> "$RESULTS"
done
echo "[auto-eval $RUN] ALL DONE $(date -u +%H:%M)" >> "$RESULTS"
