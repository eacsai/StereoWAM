#!/usr/bin/env bash
# FFS round-1 multi-GPU scheduler (#1 VLMInputFFS / #2 ControlNetFFS / #3 VLMControlNetFFS).
#
# 在【每台机器】各跑一份, 经共享 /mnt/data 队列协调 (h100a + h100b 共享 KPFS, 机器间无 ssh):
#   h100a:  MY_GPUS=0   nohup bash scripts/h100b/ffs_scheduler.sh >> .../ffs_sched_h100a.log 2>&1 &
#   h100b:  MY_GPUS="0 1" nohup bash scripts/h100b/ffs_scheduler.sh >> .../ffs_sched_h100b.log 2>&1 &
#
# 硬约束 (用户授权的全自动前提):
#   1. 前提 = B(warm-start源, cam_rope 单帧 stereo 右先左后) 的 30k ckpt 必须存在才动 (warm-start 需要它)。
#   2. 绝不 kill 任何在跑的训练; 只往 mem 空闲(<阈值)的卡上起新任务。
#   3. smoke 必须先 PASS 才排 FFS; smoke FAIL → 不排队, 记日志, 退出等用户(根本 blocker)。
#   4. 每个方法用 mkdir 原子 claim, 同一方法跨机器只起一次。
#   5. 哪张卡空了就在哪起, 依次 #1 → #2 → #3 (三卡最终并行各一个)。
set -uo pipefail

# ---- config (env 可覆盖) ----
REPO=${REPO:-/mnt/data/wangqiwei/wangqiwei/starVLA}
CKPT=${CKPT:-$REPO/playground/Checkpoints}
Q=${Q:-$CKPT/ffs_queue}
B_CKPT=${B_CKPT:-$CKPT/qwen3p5_0p8b_4suite_stereo_camrope_rightprimary_ourrender_30k/checkpoints/steps_30000_pytorch_model.pt}
LAUNCHER=${LAUNCHER:-scripts/h100b/run_qwen0p8_groot_ffs.sh}
SMOKE=${SMOKE:-scripts/h100b/smoke_groot_ffs.py}
CONDA_VENV=${CONDA_VENV:-/opt/conda/envs/starvla/bin}
PY=${PY:-$CONDA_VENV/python}
BASE_VLM=${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-0.8B}
FFS_REPO_DIR=${FFS_REPO_DIR:-/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo}
FFS_MODEL=${FFS_MODEL:-$FFS_REPO_DIR/weights/20-30-48/model_best_bp2_serialize.pth}
GPU_FREE_MB=${GPU_FREE_MB:-4000}          # mem.used < 这个 = 空闲
LAUNCH_SETTLE_S=${LAUNCH_SETTLE_S:-240}   # 起一个后等它占住卡再扫下一个(防同机重复占同卡)
POLL_S=${POLL_S:-60}
MY_GPUS="${MY_GPUS:?set MY_GPUS, e.g. MY_GPUS=0 (h100a) or MY_GPUS='0 1' (h100b)}"

HOST=$(hostname)
LOG_TAG="ffssched/$HOST"
ts(){ date -u +%FT%TZ; }
log(){ echo "[$LOG_TAG $(ts)] $*"; }

# name|framework|run_id|port  (顺序 = 排队顺序 #1 #2 #3)
METHODS=(
"vlminput|QwenGR00T_VLMInputFFS|qwen3p5_0p8b_ffs_vlminput_warmstartB_30k|29731"
"controlnet|QwenGR00T_ControlNetFFS|qwen3p5_0p8b_ffs_controlnet_warmstartB_30k|29732"
)

cd "$REPO" || { log "FATAL: REPO 不存在 $REPO"; exit 1; }
mkdir -p "$Q"
SMOKE_LOG="$CKPT/ffs_smoke_run.log"

gpu_used_mb(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -dc '0-9'; }
free_gpu(){  # echo 第一个空闲 GPU id (取自 MY_GPUS), 没有则空
  for g in $MY_GPUS; do
    local mb; mb=$(gpu_used_mb "$g")
    [ -n "$mb" ] && [ "$mb" -lt "$GPU_FREE_MB" ] && { echo "$g"; return 0; }
  done
  return 1
}
write_status(){ echo "$1" > "$Q/SMOKE_STATUS.tmp" && mv -f "$Q/SMOKE_STATUS.tmp" "$Q/SMOKE_STATUS"; }

log "=== FFS scheduler 启动 (HOST=$HOST, MY_GPUS=[$MY_GPUS], REPO=$REPO) ==="
log "前提: 等 B 30k ckpt = $B_CKPT"

# ---- Phase 1: 等 B 的 30k ckpt (warm-start 源) ----
while ! { [ -f "$B_CKPT" ] && [ "$(stat -c %s "$B_CKPT" 2>/dev/null || echo 0)" -gt 1000000 ]; }; do
  sleep "$POLL_S"
done
log "✅ B 30k ckpt 就绪"

# ---- Phase 2: smoke gate (某台机器在空闲卡上跑一次; 其它机器等 SMOKE_STATUS) ----
while [ ! -f "$Q/SMOKE_STATUS" ]; do
  g=$(free_gpu) || { sleep "$POLL_S"; continue; }
  if mkdir "$Q/smoke_claim" 2>/dev/null; then
    log "在 GPU $g 上跑 smoke (本机抢到 claim) → $SMOKE_LOG"
    if CUDA_VISIBLE_DEVICES="$g" PYTHONPATH="$REPO:$FFS_REPO_DIR:${PYTHONPATH:-}" FFS_REPO_DIR="$FFS_REPO_DIR" \
         "$PY" "$SMOKE" --framework all \
           --pretrained-ckpt "$B_CKPT" --base-vlm "$BASE_VLM" --ffs-model-path "$FFS_MODEL" \
           >> "$SMOKE_LOG" 2>&1; then
      write_status PASS; log "✅ smoke PASS"
    else
      write_status FAIL; log "⚠️ smoke FAIL — 不排队 FFS, 见 $SMOKE_LOG, 等用户"
    fi
    break
  fi
  sleep "$POLL_S"
done
# 若本机没跑 smoke, 等别的机器写出 SMOKE_STATUS
while [ ! -f "$Q/SMOKE_STATUS" ]; do sleep 30; done
if [ "$(cat "$Q/SMOKE_STATUS" 2>/dev/null)" != PASS ]; then
  log "smoke 非 PASS ($(cat "$Q/SMOKE_STATUS" 2>/dev/null)) → 退出等用户"
  exit 0
fi
log "smoke PASS — 开始按空闲卡排 #1/#2/#3"

# ---- Phase 3: 按空闲卡原子 claim + 起 FFS ----
while true; do
  remaining=0
  for m in "${METHODS[@]}"; do
    IFS='|' read -r name fw rid port <<< "$m"
    [ -d "$Q/claim_$name" ] && continue
    remaining=1
    g=$(free_gpu) || break
    if mkdir "$Q/claim_$name" 2>/dev/null; then
      # claim 到手后再确认这张卡仍空 (起任务有间隔, 防 race)
      mb=$(gpu_used_mb "$g")
      if [ -z "$mb" ] || [ "$mb" -ge "$GPU_FREE_MB" ]; then
        log "claim $name 后 GPU $g 已被占, 释放 claim 下轮重试"
        rmdir "$Q/claim_$name" 2>/dev/null || true
        break
      fi
      runlog="$CKPT/${rid}.log"
      log "起 #$name ($fw) on $HOST GPU $g | warm-start B | port $port | run_id $rid | log $runlog"
      FRAMEWORK="$fw" RUN_ID="$rid" PRETRAINED_CKPT="$B_CKPT" \
        GPUS="$g" PORT="$port" BS=32 MAX_STEPS=30000 SAVE_INTERVAL=10000 \
        FREEZE_MODULES=qwen_vl_interface NUM_PROCESSES=1 \
        BASE_VLM="$BASE_VLM" FFS_REPO_DIR="$FFS_REPO_DIR" FFS_MODEL_PATH="$FFS_MODEL" \
        nohup bash "$LAUNCHER" >> "$runlog" 2>&1 < /dev/null &
      echo "$!" > "$Q/claim_$name/pid"
      { echo "host=$HOST"; echo "gpu=$g"; echo "started=$(ts)"; echo "run_id=$rid"; echo "log=$runlog"; } > "$Q/claim_$name/info"
      log "  → $rid pid $! (GPU $g); 等 ${LAUNCH_SETTLE_S}s 占卡再扫下一个"
      sleep "$LAUNCH_SETTLE_S"
    fi
  done
  [ "$remaining" = 0 ] && { log "全部 #1/#2/#3 已 claim, $HOST scheduler 退出"; break; }
  sleep "$POLL_S"
done
