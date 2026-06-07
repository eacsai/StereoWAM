#!/bin/bash
# h100b GPU1 接力: 4D(GR00T 3帧 primary+wrist)跑完后, 在腾出的 GPU1 上依次再跑两个 GR00T 3帧 run,
# 全部用我们渲染的同一份数据 (OURRENDER_PW, 像素完全相同, 只换相机通道):
#   1. mono   = 只 primary            (libero_all_3frame_mono)
#   2. stereo = primary + right_view  (libero_all_3frame_stereo)
# => 3帧输入视角消融 (pw[4D] vs mono vs stereo), 同 action head(QwenGR00T), 同 eff128, 公平对比.
# 只等前一个跑完, 绝不 kill (feedback_no_kill).
set -uo pipefail
cd /mnt/data/wangqiwei/wangqiwei/starVLA
CKPTDIR=playground/Checkpoints
LOG=${CKPTDIR}/chain_groot_3frame_mono_stereo.log
LAUNCHER=scripts/h100b/run_qwen2p5vl3b_groot_4suite_stereo.sh
ts(){ date -u +%m-%dT%H:%M:%S; }
log(){ echo "[chain $(ts)] $*" | tee -a "$LOG"; }

wait_gone(){ # $1 = run_id; 等它的 train 进程消失(连续确认2次, 抗 ps 抖动). 按 run_id grep, 不自匹配.
  local rid="$1" gone=0
  while [ "$gone" -lt 2 ]; do
    if ps -eo cmd | grep train_starvla | grep -F -- "--run_id $rid " | grep -v grep >/dev/null 2>&1; then
      gone=0; log "  $rid 还在训练, 等待 ..."
    else
      gone=$((gone+1)); log "  $rid 进程未找到 ($gone/2)"
    fi
    sleep 60
  done
}

launch(){ # $1=run_id $2=data_mix $3=port
  local rid="$1" mix="$2" port="$3"
  log "启动 $rid (GR00T 3帧, ourrender, mix=$mix, BS16xGA8=eff128, GPU1)"
  FRAMEWORK=QwenGR00T \
  BASE_VLM=./playground/Pretrained_models/Qwen3.5-0.8B \
  DATA_ROOT=playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW \
  DATA_MIX="$mix" \
  RUN_ID="$rid" \
  BS=16 MAX_STEPS=30000 SAVE_INTERVAL=10000 \
  GPUS=1 NUM_PROCESSES=1 PORT="$port" \
  DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_ga8.yaml \
  nohup bash "$LAUNCHER" >> ${CKPTDIR}/${rid}.log 2>&1 < /dev/null &
  log "  $rid 已启动 pid $!"
  sleep 200
  if grep -qiE "out of memory|CUDA error|Traceback \(most recent" "${CKPTDIR}/${rid}.log" 2>/dev/null; then
    log "  ⚠️ ${rid}.log 200秒内有报错 — 需人工查看(不自动修); 接力仍会继续等它"
  else
    log "  $rid 200秒健康, step=$(grep -oE '[0-9]+/30000' ${CKPTDIR}/${rid}.log 2>/dev/null | tail -1)"
  fi
}

FOURD=qwen3p5_0p8b_4d3frame_primarywrist_ourrender_30k
MONO=qwen3p5_0p8b_4d3frame_monoprimary_ourrender_30k
STEREO=qwen3p5_0p8b_4d3frame_stereo_primaryright_ourrender_30k

log "=== 接力开始: 4D -> mono -> stereo (GR00T 3帧, GPU1) ==="
log "等 4D ($FOURD) 跑完 ..."
wait_gone "$FOURD"
ck="${CKPTDIR}/${FOURD}/checkpoints/steps_30000_pytorch_model.pt"
{ [ -f "$ck" ] && [ "$(stat -c %s "$ck" 2>/dev/null||echo 0)" -gt 1000000 ]; } && log "4D 正常完成 30k" || log "警告 4D 30k ckpt 缺失(可能崩了) — 仍继续起 mono"
launch "$MONO" libero_all_3frame_mono 29717

log "等 mono ($MONO) 跑完 ..."
wait_gone "$MONO"
ck="${CKPTDIR}/${MONO}/checkpoints/steps_30000_pytorch_model.pt"
{ [ -f "$ck" ] && [ "$(stat -c %s "$ck" 2>/dev/null||echo 0)" -gt 1000000 ]; } && log "mono 正常完成 30k" || log "警告 mono 30k ckpt 缺失(可能崩了) — 仍继续起 stereo"
launch "$STEREO" libero_all_3frame_stereo 29718

log "=== 接力完成: stereo 已启动(3帧 GR00T 消融的最后一个) ==="
