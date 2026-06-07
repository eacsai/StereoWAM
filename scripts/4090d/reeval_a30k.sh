#!/usr/bin/env bash
# One-off re-eval of A official step 30000 from the LOCAL ckpt on 4090d.
# h100b is down, but eval_qwen2p5vl_4suite.sh now has a local-fallback (ckpt+config+stats already local).
# Picks <=2 free GPUs (>=15GB free, idle-first), runs the 4 suites (1 server/GPU, thread-capped),
# then appends a 4/4 result block to the official RESULTS file.
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
RID=qwen2p5vl3b_4suite_primarywrist_official_90k
VK="primary,wrist"; STEP=30000
RESULTS=playground/Checkpoints/qwen2p5vl_pw_official_90k_eval_results.txt
LOG=playground/Checkpoints/reeval_a30k.log
NEED_MB=15000
ts(){ date -u +%m-%dT%H:%M:%S; }

gpus=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' -v need=$NEED_MB '($2+0)>=need{print ($3+0)" "($2+0)" "$1}' | sort -k1,1n -k2,2nr | head -2 | awk '{print $3}' | tr '\n' ' ')
echo "[reeval $(ts)] step$STEP -> dynamic GPUs [$gpus]" | tee -a "$LOG"
[ -n "$(echo $gpus | tr -d ' ')" ] || { echo "[reeval $(ts)] no GPU >=${NEED_MB}MB free, abort" | tee -a "$LOG"; exit 1; }

bash scripts/4090d/eval_qwen2p5vl_4suite.sh "$RID" "$STEP" "$VK" "$gpus" >> "$LOG" 2>&1 || echo "[reeval $(ts)] eval nonzero" | tee -a "$LOG"

edir="playground/Checkpoints/${RID}/eval_step${STEP}_$(echo "$VK" | tr ',' '_')"
ok=0; block="===== $RID step=${STEP} vk=$VK (REEVAL $(date -u)) ====="$'\n'
for s in libero_spatial libero_object libero_goal libero_10; do
  sr=$(grep 'Total success rate' "${edir}/${s}/client.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' | tail -1)
  [ -n "$sr" ] && ok=$((ok+1))
  block+="  ${s}: SR=${sr:-FAILED}"$'\n'
done
printf '%s' "$block" | tee -a "$LOG"
if [ "$ok" -ge 4 ]; then
  printf '%s' "$block" >> "$RESULTS"
  echo "[reeval $(ts)] RECORDED 4/4 to RESULTS" | tee -a "$LOG"
else
  echo "[reeval $(ts)] INCOMPLETE ${ok}/4 — NOT recorded" | tee -a "$LOG"
fi
echo "REEVAL_A30K_DONE" | tee -a "$LOG"
