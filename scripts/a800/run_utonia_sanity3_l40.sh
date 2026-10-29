#!/bin/bash
set -u

cd /home/wangqw/ICLR27/starVLA

RUN_ID=qwen0p8_groot_utonia_prompttoken_leftprimary_l40_4gpu_bs8ga4_eff128_fromscratch_0720_train
CKPT=playground/Checkpoints/$RUN_ID/checkpoints/steps_30000_pytorch_model.pt
ROOT=playground/Checkpoints/$RUN_ID/eval_sanity3_step30000_primary_left_view

if [ -e "$ROOT" ]; then
  echo "[sanity] refuse: output already exists: $ROOT"
  exit 1
fi
mkdir -p "$ROOT"

run_lane() {
  suite=$1
  gpu=$2
  port=$3
  ed="$ROOT/$suite"
  mkdir -p "$ed"
  EVAL_NUM_TRIALS=1 EVAL_MAX_TASKS=3 bash scripts/a800/eval_one_ckpt_l40.sh "$CKPT" "$gpu" "$port" "$ed" "$suite" primary,left_view 1 0 > "$ed/lane.log" 2>&1
  echo "[sanity] $suite rc=$?"
}

run_lane libero_spatial 4 6810 &
p1=$!
run_lane libero_object 5 6811 &
p2=$!
run_lane libero_goal 6 6812 &
p3=$!
run_lane libero_10 7 6813 &
p4=$!

wait "$p1"
wait "$p2"
wait "$p3"
wait "$p4"

echo "[sanity] DONE"
for suite in libero_spatial libero_object libero_goal libero_10; do
  echo "### $suite"
  grep -E "Total success rate|Total episodes|FATAL|_ARRAY_API|Traceback" "$ROOT/$suite/client.log" "$ROOT/$suite/server.log" "$ROOT/$suite/lane.log" 2>/dev/null | tail -8 || true
done
