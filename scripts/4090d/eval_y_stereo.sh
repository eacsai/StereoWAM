#!/usr/bin/env bash
# Eval Y stereo ckpt(s) on libero_goal, primary + right_view dual-image.
#
# Two key differences vs eval_z1_step10000_mono.sh (mono single-image baseline):
#   1. --args.video-keys primary,right_view --args.gripper-convention $GRIPPER_CONVENTION  → client sends 2 images per step
#   2. eval_libero.py auto-detects "right_view" in video_keys → routes env build
#      to make_stereo_env (sibling rightview camera injected via robosuite
#      set_xml_processor; survives reset). baseline default 0.06m, MUST match
#      training-time renderer (scripts/4090d/regenerate_libero_stereo.py default).
#
# Prereq: Y ckpt rsynced from jinshan_dev container to 4090d under RUN_ID dir.
# Example rsync (run BEFORE this script):
#   rsync -av jinshan_dev:/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints/goal_stereo_30k_qwenpi_0520/checkpoints/ \
#         /data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/goal_stereo_30k_qwenpi_0520/checkpoints/
#
# Usage:
#   RUN_ID=goal_stereo_30k_qwenpi_0520 STEPS=15000 bash scripts/4090d/eval_y_stereo.sh
#   # or sweep all ckpts present:
#   RUN_ID=goal_stereo_30k_qwenpi_0520 SWEEP=1 bash scripts/4090d/eval_y_stereo.sh
set -euo pipefail

RUN_ID=${RUN_ID:?must set RUN_ID (e.g. goal_stereo_30k_qwenpi_0520)}
PORT=${PORT:-6696}
GPU=${GPU:-6}
TASK_SUITE=${TASK_SUITE:-libero_goal}
STEREO_BASELINE=${STEREO_BASELINE:-0.06}
GRIPPER_CONVENTION=${GRIPPER_CONVENTION:-openvla}
NUM_TRIALS=${NUM_TRIALS:-10}

STARVLA_DIR=/data/wangqiwei/ICLR2026/starVLA
LIBERO_VENV=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python
CKPT_DIR=$STARVLA_DIR/playground/Checkpoints/$RUN_ID/checkpoints
RUN_ROOT=$STARVLA_DIR/playground/Checkpoints/$RUN_ID

if [ ! -d "$CKPT_DIR" ]; then
  echo "ABORT: $CKPT_DIR does not exist. rsync Y ckpts from container first." >&2
  exit 1
fi

# Build STEPS_LIST: explicit STEPS env var OR SWEEP=1 picks all steps_*.pt
if [ "${SWEEP:-0}" = "1" ]; then
  STEPS_LIST=$(ls $CKPT_DIR/steps_*_pytorch_model.pt 2>/dev/null \
    | sed -E "s|.*/steps_([0-9]+)_pytorch_model\.pt|\1|" | sort -n)
  [ -z "$STEPS_LIST" ] && { echo "ABORT: no steps_*_pytorch_model.pt in $CKPT_DIR"; exit 1; }
else
  STEPS_LIST=${STEPS:?must set STEPS (e.g. 15000) or SWEEP=1}
fi

echo "=== Y stereo eval: RUN_ID=$RUN_ID  steps={$STEPS_LIST}  GPU=$GPU  port=$PORT ==="
echo "                  video_keys=primary,right_view  stereo_baseline=${STEREO_BASELINE}m"

for STEPS in $STEPS_LIST; do
  CKPT=$CKPT_DIR/steps_${STEPS}_pytorch_model.pt
  EVAL_DIR=$RUN_ROOT/eval_logs/step${STEPS}_stereo

  [ ! -f "$CKPT" ] && { echo "SKIP: $CKPT not found"; continue; }
  if [ -d "$EVAL_DIR/videos" ] && [ "$(ls $EVAL_DIR/videos 2>/dev/null | wc -l)" -ge $((NUM_TRIALS * 10)) ]; then
    echo "SKIP: $EVAL_DIR already has $((NUM_TRIALS * 10))+ episodes"; continue
  fi
  mkdir -p $EVAL_DIR/videos
  echo "--- step $STEPS -> $EVAL_DIR ($(date)) ---"

  pkill -9 -f "server_policy.py.*--port $PORT" 2>/dev/null || true
  sleep 3

  cd $STARVLA_DIR
  echo "  [server] launching GPU=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python deployment/model_server/server_policy.py \
    --ckpt_path $CKPT --port $PORT --use_bf16 \
    > $EVAL_DIR/server.log 2>&1 &
  SERVER_PID=$!

  for i in $(seq 1 60); do
    if grep -q "listening" $EVAL_DIR/server.log 2>/dev/null; then
      echo "  [server ready, PID=$SERVER_PID]"; break
    fi
    sleep 2
  done

  echo "  [client] launching (video_keys=primary,right_view)"
  export LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO
  export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
  export PYTHONPATH=${LIBERO_HOME}:$STARVLA_DIR:${PYTHONPATH:-}
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  $LIBERO_VENV examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path $CKPT --args.host 127.0.0.1 --args.port $PORT \
    --args.task-suite-name $TASK_SUITE --args.num-trials-per-task $NUM_TRIALS --args.max-tasks -1 \
    --args.video-out-path $EVAL_DIR/videos \
    --args.video-keys primary,right_view --args.gripper-convention $GRIPPER_CONVENTION \
    --args.stereo-baseline $STEREO_BASELINE \
    > $EVAL_DIR/client.log 2>&1 || true

  SUCC=$(ls $EVAL_DIR/videos 2>/dev/null | grep -c success || true)
  FAIL=$(ls $EVAL_DIR/videos 2>/dev/null | grep -c failure || true)
  TOTAL=$((SUCC + FAIL))
  echo "  [done] step $STEPS stereo: $SUCC/$TOTAL success ($(date))"

  kill -9 $SERVER_PID 2>/dev/null || true
  sleep 3
done

echo "=== Y STEREO EVAL SWEEP DONE $(date) ==="
