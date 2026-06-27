#!/usr/bin/env bash
# Auto-eval daemon (runs ON 4090d): watch ONE a800 leftprimary training for its step-30000
# (final) ckpt, pull it to 4090d, eval all 4 LIBERO suites, record per-suite SR + mean.
#
# Adapted from the battle-tested auto_eval_0p8b_30k_run.sh. Two changes:
#   (1) ckpt SOURCE = a800 (not h100b) — pulled via the push key over a800's ssh port.
#   (2) we PRE-STAGE ckpt+config+stats into the 4090d local run dir, then call
#       eval_qwen2p5vl_4suite.sh which (h100b has no leftprimary run -> stat fails ->
#       its built-in "use existing LOCAL copy" fallback) evals the local ckpt.
# video_keys are AUTO-DERIVED from the ckpt config (leftprimary -> primary,left_view) inside
# eval_qwen2p5vl_4suite.sh / eval_one_ckpt.sh; we also pass VK explicitly as a SAFETY ASSERTION
# (4suite FATALs if derived != passed), so a mis-saved non-leftprimary config fails loud, never silent.
#
# flock-serialized with all other eval daemons (one eval at a time on shared 4090d) + dynamic
# free-GPU pick + retry-until-4/4 + idempotent dedup (recorded step is skipped) + fail-closed.
#
# Usage: auto_eval_30k_a800.sh <RUN_ID> <A800_IP> <A800_PORT>
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA

RID="${1:?need RUN_ID}"; SRC_IP="${2:?need a800 ip}"; SRC_PORT="${3:?need a800 ssh port}"
STEP="${STEP:-30000}"
VK="primary,left_view"                                   # leftprimary convention (asserted vs config-derived)
EVAL=scripts/4090d/eval_qwen2p5vl_4suite.sh
RESULTS="playground/Checkpoints/${RID}_eval_results.txt"
REGISTRY="docs/experiments/leftprimary_30k_autoeval.md"
LOG="playground/Checkpoints/auto_eval_${RID}.log"
LOCK="playground/Checkpoints/.eval_4090d.lock"           # SHARED by all eval daemons
A800="ssh -i /data/wangqiwei/.ssh/id_a800_push -p ${SRC_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=15 wangqiwei@${SRC_IP}"
A800_RUN="/home/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/${RID}"
LOCAL_RUN="playground/Checkpoints/${RID}"
CKPT="${LOCAL_RUN}/checkpoints/steps_${STEP}_pytorch_model.pt"
EDIR="${LOCAL_RUN}/eval_step${STEP}_$(echo "$VK" | tr ',' '_')"
NEED_MB="${NEED_MB:-15000}"; MAXGPU="${MAXGPU:-2}"; MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"; POLL="${POLL:-300}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

log(){ echo "[autoeval $(date -u +%m-%dT%H:%M:%S) ${RID}] $*" | tee -a "$LOG"; }
pick_eval_gpus(){ nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' -v need="$NEED_MB" '($2+0)>=need{print ($3+0)" "($2+0)" "$1}' \
  | sort -k1,1n -k2,2nr | head -"$MAXGPU" | awk '{print $3}' | tr '\n' ' '; }

touch "$LOCK" 2>/dev/null || true
mkdir -p "${LOCAL_RUN}/checkpoints" docs/experiments
log "=== daemon start src=${SRC_IP}:${SRC_PORT} step=$STEP vk=$VK ==="

# idempotent: full recorded block already present -> done
if grep -A5 -E "^===== ${RID} step=${STEP} " "$RESULTS" 2>/dev/null | grep -q "libero_10: SR="; then
  log "step$STEP already fully recorded — exit"; exit 0
fi

# 1. wait for the final ckpt on a800
remote_ck="${A800_RUN}/checkpoints/steps_${STEP}_pytorch_model.pt"
while true; do
  sz=$($A800 "stat -c %s ${remote_ck}" 2>/dev/null | tr -dc 0-9)
  [ -n "${sz:-}" ] && [ "${sz:-0}" -gt 1000000 ] && break
  log "waiting for step$STEP ckpt on a800 (poll ${POLL}s) ..."; sleep "$POLL"
done
log "step$STEP ckpt ready on a800 (size $((sz/1000000))MB)"

# 2. pre-stage ckpt + config.full + stats -> 4090d local (so 4suite uses the local copy)
LOCAL_SZ=$( [ -f "$CKPT" ] && stat -c %s "$CKPT" || echo 0 )
if [ "$LOCAL_SZ" != "$sz" ]; then
  log "pulling ckpt+config+stats from a800 ($((sz/1000000))MB) ..."
  $A800 "cat ${remote_ck}" > "$CKPT"
  $A800 "cat ${A800_RUN}/config.full.yaml" > "${LOCAL_RUN}/config.full.yaml"
  $A800 "cat ${A800_RUN}/dataset_statistics.json" > "${LOCAL_RUN}/dataset_statistics.json"
  cp "${LOCAL_RUN}/config.full.yaml" "${LOCAL_RUN}/config.yaml"
  NEW_SZ=$(stat -c %s "$CKPT")
  [ "$NEW_SZ" = "$sz" ] || { log "[FATAL] pulled ckpt size mismatch local=$NEW_SZ remote=$sz"; rm -f "$CKPT"; exit 2; }
  log "pull OK"
else
  log "ckpt already local + size matches"
fi

# 3. eval 4 suites under shared flock; retry until all 4 SR numeric
recorded=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  exec 9>"$LOCK"
  if ! flock -w 7200 9; then log "lock wait timeout a$attempt"; exec 9>&-; sleep 60; continue; fi
  gpus=""
  while true; do gpus=$(pick_eval_gpus); [ -n "$(echo "$gpus" | tr -d ' ')" ] && break
    log "no GPU with >=${NEED_MB}MB free — wait 120s (holding lock)"; sleep 120; done
  log "attempt$attempt -> GPUs [$gpus] : eval_4suite (vk=$VK asserted vs config-derived)"
  bash "$EVAL" "$RID" "$STEP" "$VK" "$gpus" >> "$LOG" 2>&1 || log "WARN eval_4suite nonzero a$attempt"
  flock -u 9; exec 9>&-
  ok=0; sum=0; block="===== ${RID} step=${STEP} vk=${VK} ($(date -u)) ====="$'\n'
  for s in "${SUITES[@]}"; do
    sr=$(grep -iE "Total success rate" "${EDIR}/${s}/client.log" 2>/dev/null | grep -oE "[0-9]+\.?[0-9]*" | tail -1)
    if [ -n "$sr" ]; then ok=$((ok+1)); sum=$(awk -v a="$sum" -v b="$sr" 'BEGIN{print a+b}'); fi
    block+="  ${s}: SR=${sr:-FAILED}"$'\n'
  done
  if [ "$ok" -ge 4 ]; then
    mean=$(awk -v s="$sum" 'BEGIN{printf "%.4f", s/4}')
    block+="  MEAN: ${mean}"$'\n'
    printf '%s' "$block" | tee -a "$LOG"
    printf '%s' "$block" >> "$RESULTS"
    printf '%s' "$block" >> "$REGISTRY"
    log "step$STEP RECORDED (4/4, mean=$mean, attempt$attempt)"
    rm -f "$CKPT"; log "deleted local ckpt copy (a800 keeps the real one)"
    recorded=1; break
  fi
  log "step$STEP INCOMPLETE ($ok/4) attempt$attempt — retry"; sleep 60
done
[ "$recorded" = 1 ] || { log "[FAIL] step$STEP not recorded after $MAX_ATTEMPTS attempts (fail-closed, ckpt kept for debug)"; exit 1; }
log "=== daemon done RID=$RID ==="
