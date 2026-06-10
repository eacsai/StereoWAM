#!/usr/bin/env bash
# #4 injection-modality ablation scheduler (SMOKE-GATED).
#
# 3 from-scratch arms (all eff128 = BS32 x GA4, same as B/fullinject):
# All 3 arms run CAM_ROPE=0: the cam_rope patch is inert (zero-init dead-lock) but its
# d_c branch breaks FlashAttention's 256 head_dim limit -> 4x slower full finetune.
# Bypass is output-equivalent (smoke_camrope_disable_equivalence.py proves it on GPU).
#   keep-64  : depth tokens kept in the action-head cross-attn (GR00T default)
#   strip-64 : depth tokens stripped before the action head (QwenPI-style gather)
#   depthimage: FFS disparity rendered as a 3rd turbo image (cam_id=1 = left pose)
#
# Phase 1 (gate): whichever machine frees a GPU first runs BOTH integration smokes on it
#   (smoke_depthtoken_strip + smoke_depthimage). SMOKE_ALL_PASS -> proceed; else HALT (launch
#   nothing) + leave a marker. Phase 2: launch the 3 arms as GPUs free.
#
# Per-machine (run on h100a MY_GPUS=0 AND h100b MY_GPUS="0 1"); shared /mnt/data queue with
# atomic mkdir claims. NEVER kills a running job; only uses mem-free GPUs.
set -uo pipefail
REPO=/mnt/data/wangqiwei/wangqiwei/starVLA
cd "$REPO"
CKPT=$REPO/playground/Checkpoints
Q=$CKPT/ffs4_ablation_queue
LAUNCHER=scripts/h100b/run_qwen0p8_groot_ffs.sh
FFS_REPO=/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo
PY=/opt/conda/envs/starvla/bin/python3.10
DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml   # GA4 -> eff128 with BS32 (proven config; BS64 hit FFS grid_sample cuDNN NOT_SUPPORTED bug at step 0)
GPU_FREE_MB=${GPU_FREE_MB:-4000}
MY_GPUS=${MY_GPUS:-"0 1"}
SETTLE=${SETTLE:-300}
LOG=$CKPT/ffs4_ablation_scheduler.log
SMOKE_PASS=$Q/SMOKE_PASSED
SMOKE_FAIL=$Q/SMOKE_FAILED
ts(){ date -u +%FT%TZ; }
log(){ echo "[ffs4 $(ts)] $*" | tee -a "$LOG"; }
mkdir -p "$Q"
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -dc 0-9; }
free_gpu(){ for g in $MY_GPUS; do local mb; mb=$(gpu_used "$g"); [ -n "$mb" ] && [ "$mb" -lt "$GPU_FREE_MB" ] && { echo "$g"; return 0; }; done; return 1; }

# prio|framework|run_id|strip|ntok|pool|port
# 64 depth tokens (pool 8x8): user's choice (reverted from the 16 experiment). The 16<->64 token
# count was NOT the slow-training cause (16 still spiked to 30s) -> the real cause is per-step
# host-device syncs in the insertion hook (fixed separately). With the hook fix 64 is fast enough;
# 64 vs 16 only differs by O(seq^2) attention on +64 vs +16 tokens (small). keep/strip MUST share
# the same count for the action-head-access comparison to stay single-variable.
QUEUE=(
  "1|QwenGR00T_DepthTokenFFS|qwen3p5_0p8b_ffs_depthtoken_keep_fromscratch_30k|0|64|8|29736"
  "2|QwenGR00T_DepthTokenFFS|qwen3p5_0p8b_ffs_depthtoken_strip_fromscratch_30k|1|64|8|29737"
  "3|QwenGR00T_DepthImageFFS|qwen3p5_0p8b_ffs_depthimage_fromscratch_30k|0|16|4|29738"
)

run_smokes(){   # gpu -> echo PASS|FAIL ; full integration smokes on a real GPU
  local g="$1" ok=1 s
  # log goes to stderr here: this function's stdout is command-substituted by the
  # caller, so any log line on stdout would corrupt the PASS/FAIL verdict string.
  log "  smoke-gate on GPU $g (equivalence + depthtoken_strip + depthimage; ~few min) ..." >&2
  for s in smoke_depthtoken_strip smoke_depthimage smoke_camrope_disable_equivalence; do
    local slog="$CKPT/ffs4_${s}.log"
    if CUDA_VISIBLE_DEVICES="$g" PYTHONPATH="$REPO:$FFS_REPO" "$PY" "scripts/h100b/$s.py" --device cuda > "$slog" 2>&1 && grep -q SMOKE_ALL_PASS "$slog"; then
      log "    $s PASS" >&2
    else
      ok=0; log "    $s FAIL (see $slog)" >&2
    fi
  done
  [ "$ok" = 1 ] && echo PASS || echo FAIL
}

launch_one(){   # prio fw rid strip ntok pool port gpu
  local prio="$1" fw="$2" rid="$3" strip="$4" ntok="$5" pool="$6" port="$7" g="$8"
  log "起 #4-P$prio ($fw -> $rid) on GPU $g | BS32 x GA4 = eff128 | strip=$strip ntok=$ntok pool=$pool | CAM_ROPE=0 (inert-bypass, FA2 fast path)"
  FRAMEWORK="$fw" RUN_ID="$rid" PRETRAINED_CKPT="" FREEZE_MODULES="" \
    STRIP_DEPTH="$strip" NUM_DEPTH_TOKENS="$ntok" POOL_HW="$pool" CAM_ROPE=0 \
    GPUS="$g" PORT="$port" BS=32 DS_CONFIG="$DS_CONFIG" \
    MAX_STEPS=30000 SAVE_INTERVAL=10000 NUM_PROCESSES=1 FFS_REPO_DIR="$FFS_REPO" \
    setsid nohup bash "$LAUNCHER" >> "$CKPT/${rid}.log" 2>&1 < /dev/null &
  local launcher_pid=$!
  log "  pid ${launcher_pid} (GPU $g); 等 ${SETTLE}s 占卡再扫下一个"
  # The claim dir is permanent: if the launcher dies immediately (guard/preflight
  # refusal), the arm would be silently skipped forever. Verify it survived a beat.
  sleep 10
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    log "  ⚠️ FATAL: launcher pid ${launcher_pid} for $rid died within 10s (guard/preflight refusal?) — claim kept to avoid relaunch loops; inspect $CKPT/${rid}.log and rmdir the claim manually to retry"
  fi
}

log "=== #4 ablation scheduler 启动 (MY_GPUS=[$MY_GPUS]; ${#QUEUE[@]} arms; BS32 GA4 eff128) ==="

# ----- Phase 1: smoke gate (run once, whoever grabs the first free GPU) -----
while [ ! -f "$SMOKE_PASS" ] && [ ! -f "$SMOKE_FAIL" ]; do
  if mkdir "$Q/claim_smoke" 2>/dev/null; then
    g=$(free_gpu) || { rmdir "$Q/claim_smoke" 2>/dev/null; sleep 60; continue; }
    log "  claimed smoke-gate on GPU $g"
    res=$(run_smokes "$g")
    if [ "$res" = PASS ]; then touch "$SMOKE_PASS"; log "  ✅ SMOKE GATE PASSED -> 放行 3 个 arm"; else touch "$SMOKE_FAIL"; log "  ❌ SMOKE GATE FAILED -> 不起任何训练, 停下 (查 ffs4_*.log)"; fi
  else
    sleep 30   # another scheduler is running the smoke
  fi
done
[ -f "$SMOKE_FAIL" ] && { log "=== smoke 挂了, scheduler 退出 (未起训练) ==="; exit 1; }

# ----- Phase 2: launch arms as GPUs free -----
while true; do
  remaining=0
  for row in "${QUEUE[@]}"; do
    IFS='|' read -r prio fw rid strip ntok pool port <<< "$row"
    [ -d "$Q/claim_$rid" ] && continue
    remaining=1
    g=$(free_gpu) || continue
    mkdir "$Q/claim_$rid" 2>/dev/null || continue
    launch_one "$prio" "$fw" "$rid" "$strip" "$ntok" "$pool" "$port" "$g"
    sleep "$SETTLE"
  done
  [ "$remaining" = 0 ] && { log "=== #4 队列清空, scheduler 退出 ==="; break; }
  sleep 60
done
