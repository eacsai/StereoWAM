#!/bin/bash
# right-first 3帧 stereo 接力: 等 GPU1 的 interval0p5s 跑完 -> 在 GPU1 起「3帧 stereo 右先左后」
# (GR00T 3帧, video_keys=[right_view, primary], ourrender)。对比已完成的左先右后 3帧 stereo
# (qwen3p5_0p8b_4d3frame_stereo_primaryright_ourrender_30k, 保留作对比)。只等不杀 (feedback_no_kill)。
set -uo pipefail
cd /mnt/data/wangqiwei/wangqiwei/starVLA
CKPTDIR=playground/Checkpoints
LOG=${CKPTDIR}/chain_stereo_rightprimary_3frame.log
WAITRUN=qwen3p5_0p8b_4d3frame_interval0p5s_primarywrist_ourrender_30k
NEWRUN=qwen3p5_0p8b_4d3frame_stereo_rightprimary_ourrender_30k
ts(){ date -u +%m-%dT%H:%M:%S; }
log(){ echo "[chain $(ts)] $*" | tee -a "$LOG"; }

log "=== 等 GPU1 interval0p5s ($WAITRUN) 跑完 → 起 3帧 stereo 右先左后 ==="
gone=0
while [ "$gone" -lt 2 ]; do
  if ps -eo cmd | grep train_starvla | grep -F -- "--run_id $WAITRUN " | grep -v grep >/dev/null 2>&1; then
    gone=0; log "  $WAITRUN 还在训练, 等待 ..."
  else gone=$((gone+1)); log "  $WAITRUN 进程未找到 ($gone/2)"; fi
  sleep 60
done
ck="${CKPTDIR}/${WAITRUN}/checkpoints/steps_30000_pytorch_model.pt"
{ [ -f "$ck" ] && [ "$(stat -c %s "$ck" 2>/dev/null||echo 0)" -gt 1000000 ]; } && log "interval0p5s 完成 30k" || log "警告 interval0p5s 30k ckpt 缺失 — 仍起 3帧 stereo 右先左后"

log "启动 3帧 stereo 右先左后 ($NEWRUN, GR00T 3帧 [right_view,primary] ourrender, GPU1, BS16xGA8 eff128)"
FRAMEWORK=QwenGR00T BASE_VLM=./playground/Pretrained_models/Qwen3.5-0.8B \
DATA_ROOT=playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW DATA_MIX=libero_all_3frame_stereo_rightprimary \
RUN_ID=$NEWRUN BS=16 MAX_STEPS=30000 SAVE_INTERVAL=10000 \
GPUS=1 NUM_PROCESSES=1 PORT=29724 CAM_ROPE=false DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_ga8.yaml \
nohup bash scripts/h100b/run_qwen2p5vl3b_groot_4suite_stereo.sh >> ${CKPTDIR}/${NEWRUN}.log 2>&1 < /dev/null &
log "  $NEWRUN pid $!"
sleep 200
if grep -qiE "out of memory|Traceback \(most recent|CUDA error" "${CKPTDIR}/${NEWRUN}.log" 2>/dev/null; then
  log "  ⚠️ ${NEWRUN}.log 200s 有报错 — 人工查"
else log "  健康 step=$(grep -oE '[0-9]+/30000' ${CKPTDIR}/${NEWRUN}.log 2>/dev/null|tail -1)"; fi
log "=== 3帧 stereo 右先左后 已起 (对比左先右后 3帧 stereo) ==="
