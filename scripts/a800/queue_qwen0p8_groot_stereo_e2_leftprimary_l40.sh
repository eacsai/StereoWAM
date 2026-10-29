#!/bin/bash
set -u

cd /home/wangqw/ICLR27/starVLA

UTONIA_RUN_ID=qwen0p8_groot_utonia_prompttoken_leftprimary_l40_4gpu_bs8ga4_eff128_fromscratch_0720_train
E2_RUN_ID=qwen0p8_groot_stereo_cambranch_e2_bidirectional_leftprimary_l40_4gpu_bs8ga4_eff128_seed42_fromscratch_0720
E2_OUTPUT=playground/Checkpoints/$E2_RUN_ID
QUEUE_LOG=playground/Checkpoints/queue_qwen0p8_groot_stereo_e2_leftprimary_l40.log

mkdir -p playground/Checkpoints
exec >>"$QUEUE_LOG" 2>&1

echo "[queue] $(date -Is) waiting for Utonia run and GPUs 4-7"
echo "[queue] Utonia run_id=$UTONIA_RUN_ID"
echo "[queue] E2 run_id=$E2_RUN_ID"

while true; do
  ut_status="$(ps -eo pid=,args= | awk -v needle="$UTONIA_RUN_ID" '$0 !~ /awk -v needle/ && index($0, needle) {print}')"
  gpu_status="$(nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  if awk -F, 'NF >= 2 {gsub(/[[:space:]]/, "", $2); if ($2+0 >= 2000) bad=1} END {exit !(NR == 4 && !bad)}' <<< "$gpu_status"; then
    gpu_ready=1
  else
    gpu_ready=0
  fi

  if [ -z "$ut_status" ] && [ "$gpu_ready" -eq 1 ]; then
    break
  fi
  echo "[queue] $(date -Is) still waiting: utonia_active=$([ -n "$ut_status" ] && echo 1 || echo 0) gpu_ready=$gpu_ready"
  sleep 60
done

if [ -e "$E2_OUTPUT" ]; then
  echo "[queue] refuse: E2 output directory already exists: $E2_OUTPUT"
  exit 1
fi

echo "[queue] $(date -Is) starting E2 on GPUs 4-7"
GPUS=4,5,6,7 NUM_PROCESSES=4 PORT=29772 BS=8 MAX_STEPS=30000 SAVE_INTERVAL=2000 WARMUP_STEPS=5000 DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml DATA_ROOT=/home/wangqw/ICLR27/starVLA/playground/Datasets/LEROBOT_LIBERO_STEREO_LEFTPRIMARY DATA_MIX=libero_all_sfstereo_leftprimary CAM_ROPE=0 CAM_BRANCH=1 RUN_ID="$E2_RUN_ID" RESUME=0 bash scripts/a800/run_qwen0p8_groot_stereo_e2_leftprimary_l40.sh
status=$?
echo "[queue] $(date -Is) E2_TRAIN_EXIT=$status"
exit "$status"
