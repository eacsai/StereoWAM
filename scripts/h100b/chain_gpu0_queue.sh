#!/bin/bash
# h100b GPU0 完整队列接力: 当前 official QwenPI 单帧(在跑)完后, 在 GPU0 依次:
#   ① QwenPI 3帧 pw (我们渲染, 真多帧)    libero_all_3frame   FRAMEWORK=QwenPI    BS16/GA8 eff128
#   ② GR00T 单帧 mono (只 primary)        libero_all_mono     FRAMEWORK=QwenGR00T BS32/GA4 eff128
#   ③ GR00T 单帧 stereo (primary+right)   libero_all_sfstereo   FRAMEWORK=QwenGR00T BS32/GA4 eff128
# 全 ourrender (OURRENDER_PW), 30k, save每10k. 只等前一个完, 绝不 kill (feedback_no_kill).
# 端口避开 GPU1 接力(4D 29715 / GPU1-mono 29717 / GPU1-stereo 29718): GPU0 用 29716/29720/29721.
set -uo pipefail
cd /mnt/data/wangqiwei/wangqiwei/starVLA
CKPTDIR=playground/Checkpoints
LOG=${CKPTDIR}/chain_gpu0_queue.log
LAUNCHER=scripts/h100b/run_qwen2p5vl3b_groot_4suite_stereo.sh
GA8=starVLA/config/deepseeds/deepspeed_zero2_ga8.yaml
GA4=starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml
ts(){ date -u +%m-%dT%H:%M:%S; }
log(){ echo "[chain $(ts)] $*" | tee -a "$LOG"; }

wait_gone(){ # $1=run_id; 等其 train 进程消失(连续2次确认). 按完整 run_id grep, 不自匹配/不误匹配 GPU1。
  local rid="$1" gone=0
  while [ "$gone" -lt 2 ]; do
    if ps -eo cmd | grep train_starvla | grep -F -- "--run_id $rid " | grep -v grep >/dev/null 2>&1; then
      gone=0; log "  $rid 还在训练, 等待 ..."
    else gone=$((gone+1)); log "  $rid 进程未找到 ($gone/2)"; fi
    sleep 60
  done
}

launch(){ # $1=run_id $2=framework $3=data_mix $4=bs $5=ds_config $6=port
  local rid="$1" fw="$2" mix="$3" bs="$4" ds="$5" port="$6"
  log "启动 $rid (fw=$fw mix=$mix BS=$bs GPU0 port=$port)"
  FRAMEWORK="$fw" BASE_VLM=./playground/Pretrained_models/Qwen3.5-0.8B \
  DATA_ROOT=playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW DATA_MIX="$mix" \
  RUN_ID="$rid" BS="$bs" MAX_STEPS=30000 SAVE_INTERVAL=10000 \
  GPUS=0 NUM_PROCESSES=1 PORT="$port" DS_CONFIG="$ds" \
  nohup bash "$LAUNCHER" >> ${CKPTDIR}/${rid}.log 2>&1 < /dev/null &
  log "  $rid pid $!"
  sleep 200
  if grep -qiE "out of memory|CUDA error|Traceback \(most recent" "${CKPTDIR}/${rid}.log" 2>/dev/null; then
    log "  ⚠️ ${rid}.log 200s 内有报错 — 需人工查(不自动修); 接力继续等它"
  else log "  $rid 200s 健康, step=$(grep -oE '[0-9]+/30000' ${CKPTDIR}/${rid}.log 2>/dev/null|tail -1)"; fi
}

OFFICIAL=qwen3p5_0p8b_qwenpi_4suite_primarywrist_official_30k

log "=== GPU0 队列接力开始: 等 official QwenPI 单帧 ($OFFICIAL) 完 ==="
wait_gone "$OFFICIAL"
ck="${CKPTDIR}/${OFFICIAL}/checkpoints/steps_30000_pytorch_model.pt"
{ [ -f "$ck" ] && [ "$(stat -c %s "$ck" 2>/dev/null||echo 0)" -gt 1000000 ]; } && log "official 完成 30k" || log "警告 official 30k ckpt 缺失(崩?) — 仍继续起 QwenPI 3帧"

# ① QwenPI 3帧 pw (真多帧)
launch qwen3p5_0p8b_qwenpi_4d3frame_primarywrist_ourrender_30k QwenPI    libero_all_3frame 16 "$GA8" 29716
wait_gone qwen3p5_0p8b_qwenpi_4d3frame_primarywrist_ourrender_30k

# ② GR00T 单帧 mono
launch qwen3p5_0p8b_4suite_monoprimary_ourrender_30k          QwenGR00T libero_all_mono   32 "$GA4" 29720
wait_gone qwen3p5_0p8b_4suite_monoprimary_ourrender_30k

# ③ GR00T 单帧 stereo
launch qwen3p5_0p8b_4suite_stereo_primaryright_ourrender_30k  QwenGR00T libero_all_sfstereo 32 "$GA4" 29721

log "=== GPU0 队列完成: 单帧 stereo 已起(最后一个) ==="
