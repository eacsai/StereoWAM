#!/usr/bin/env bash
# Re-run ALL 6 FFS runs with the eval-mode fix (+ the other codex fixes).
# Reason: the frozen FoundationStereo was being run in TRAIN mode (36 BN + 57
# Dropout) -> the injected net[0] disparity was non-deterministic during training
# (smoke: train-mode 171x noisier than eval-mode). Every FFS run trained on
# corrupted features, so all are re-run with self.ffs.eval() asserted.
#
# Per-machine scheduler (run on h100a MY_GPUS=0 AND h100b MY_GPUS="0 1"); they
# coordinate through the shared /mnt/data queue with atomic mkdir claims. Each run
# archives its old (buggy) output dir before launching, then reuses the same run_id.
# Never kills a running job; only uses mem-free GPUs. Priority order: the clean
# from-scratch full-injection test first, cam-frozen ablations last.
set -uo pipefail
REPO=/mnt/data/wangqiwei/wangqiwei/starVLA
cd "$REPO"
CKPT=$REPO/playground/Checkpoints
Q=$CKPT/rerun_evalfix_queue
ARCHIVE=$CKPT/_prebugfix_archive
B_CKPT=$CKPT/qwen3p5_0p8b_4suite_stereo_camrope_rightprimary_ourrender_30k/checkpoints/steps_30000_pytorch_model.pt
LAUNCHER=scripts/h100b/run_qwen0p8_groot_ffs.sh
FFS_REPO=/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo
GPU_FREE_MB=${GPU_FREE_MB:-4000}
MY_GPUS=${MY_GPUS:-"0 1"}
SETTLE=${SETTLE:-240}
LOG=$CKPT/rerun_evalfix_scheduler.log
ts(){ date -u +%FT%TZ; }
log(){ echo "[rerun $(ts)] $*" | tee -a "$LOG"; }
mkdir -p "$Q" "$ARCHIVE"
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -dc 0-9; }
free_gpu(){ for g in $MY_GPUS; do local mb; mb=$(gpu_used "$g"); [ -n "$mb" ] && [ "$mb" -lt "$GPU_FREE_MB" ] && { echo "$g"; return 0; }; done; return 1; }

# priority|framework|run_id|pretrained|freeze|gate_init|port
QUEUE=(
  "1|QwenGR00T_VLMInputFFS|qwen3p5_0p8b_ffs_vlminput_fromscratch_fullinject_30k|||identity|29735"
  "2|QwenGR00T_VLMInputFFS|qwen3p5_0p8b_ffs_vlminput_warmstartB_30k|${B_CKPT}|qwen_vl_interface|zero|29731"
  "3|QwenGR00T_ControlNetFFS|qwen3p5_0p8b_ffs_controlnet_warmstartB_30k|${B_CKPT}|qwen_vl_interface|zero|29732"
  "5|QwenGR00T_ControlNetFFS|qwen3p5_0p8b_ffs_controlnet_warmstartB_camfrozen_30k|${B_CKPT}|qwen_vl_interface,stereo_cam_rope_layers|zero|29743"
)

archive_old(){   # run_id — move a prior (buggy) output dir aside so the launcher's
                 # refuse-guard passes and the run_id is reused cleanly.
  local rid="$1" d="$CKPT/$1"
  [ -d "$d" ] || return 0
  local dst="$ARCHIVE/${rid}.prebugfix_$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$d" "$dst" && log "  archived old $rid -> $dst"
}

launch_one(){   # prio fw rid pre frz gate port gpu
  local prio="$1" fw="$2" rid="$3" pre="$4" frz="$5" gate="$6" port="$7" g="$8"
  archive_old "$rid"
  log "起 P$prio ($fw -> $rid) on GPU $g | pretrained='${pre:-<none>}' freeze='${frz:-<none>}' gate=$gate"
  FRAMEWORK="$fw" RUN_ID="$rid" PRETRAINED_CKPT="$pre" FREEZE_MODULES="$frz" INJECT_GATE_INIT="$gate" \
    GPUS="$g" PORT="$port" BS=32 MAX_STEPS=30000 SAVE_INTERVAL=10000 NUM_PROCESSES=1 \
    FFS_REPO_DIR="$FFS_REPO" \
    setsid nohup bash "$LAUNCHER" >> "$CKPT/${rid}.log" 2>&1 < /dev/null &
  log "  pid $! (GPU $g); 等 ${SETTLE}s 占卡再扫下一个"
}

log "=== rerun-evalfix scheduler 启动 (MY_GPUS=[$MY_GPUS]; ${#QUEUE[@]} 项) ==="
[ -f "$B_CKPT" ] || { log "FATAL: B ckpt 不在 $B_CKPT"; exit 1; }

while true; do
  remaining=0
  for row in "${QUEUE[@]}"; do
    IFS='|' read -r prio fw rid pre frz gate port <<< "$row"
    [ -d "$Q/claim_$rid" ] && continue
    remaining=1
    g=$(free_gpu) || continue
    mkdir "$Q/claim_$rid" 2>/dev/null || continue
    launch_one "$prio" "$fw" "$rid" "$pre" "$frz" "$gate" "$port" "$g"
    sleep "$SETTLE"
  done
  [ "$remaining" = 0 ] && { log "=== 队列清空, scheduler 退出 ==="; break; }
  sleep 60
done
