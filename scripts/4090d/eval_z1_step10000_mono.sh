#!/usr/bin/env bash
# Eval Z1 (goal_only_30k_mono_primary_official_0520) step 10000 ckpt on libero_goal,
# **single primary image only** (no wrist) — matches the training-time video_keys
# from Libero4in1MonoPrimaryDataConfig (video.primary_image only).
#
# Why this script exists separately: eval_ckpt_sweep.sh (the wrist+primary baseline)
# does NOT pass --args.video-keys, so eval_libero.py falls back to its default
# "primary,wrist" → sends 2 images to the model, but Z1 only ever saw 1 image
# during training → input shape mismatch / poor results. This launcher fixes that
# by passing --args.video-keys primary explicitly.
set -euo pipefail

STEPS=10000
CKPT_DIR=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/goal_only_30k_mono_primary_official_0520/checkpoints
RUN_ROOT=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/goal_only_30k_mono_primary_official_0520
STARVLA_DIR=/data/wangqiwei/ICLR2026/starVLA
LIBERO_VENV=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python
PORT=${PORT:-6695}
GPU=${GPU:-6}
TASK_SUITE=libero_goal

CKPT=$CKPT_DIR/steps_${STEPS}_pytorch_model.pt
EVAL_DIR=$RUN_ROOT/eval_logs/step${STEPS}_mono_primary

[ ! -f "$CKPT" ] && { echo "ABORT: $CKPT not found"; exit 1; }
mkdir -p $EVAL_DIR/videos

echo "=== eval Z1 steps_${STEPS} MONO PRIMARY ONLY -> $EVAL_DIR ($(date)) ==="
echo "    GPU=$GPU port=$PORT video_keys=primary (single image)"

pkill -9 -f "server_policy.py.*--port $PORT" 2>/dev/null || true
sleep 3

cd $STARVLA_DIR
echo "  [server] launching GPU=$GPU port=$PORT"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python deployment/model_server/server_policy.py \
  --ckpt_path $CKPT --port $PORT --use_bf16 \
  > $EVAL_DIR/server.log 2>&1 &
SERVER_PID=$!
echo "  [server PID=$SERVER_PID]"

for i in $(seq 1 60); do
  if grep -q "listening" $EVAL_DIR/server.log 2>/dev/null; then
    echo "  [server ready at iter $i]"; break
  fi
  sleep 2
done

echo "  [client] launching (video_keys=primary, single image)"
export LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:$STARVLA_DIR:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
$LIBERO_VENV examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path $CKPT --args.host 127.0.0.1 --args.port $PORT \
  --args.task-suite-name $TASK_SUITE --args.num-trials-per-task 10 --args.max-tasks -1 \
  --args.video-out-path $EVAL_DIR/videos \
  --args.video-keys primary \
  > $EVAL_DIR/client.log 2>&1 || true

SUCC=$(ls $EVAL_DIR/videos 2>/dev/null | grep -c success || echo 0)
FAIL=$(ls $EVAL_DIR/videos 2>/dev/null | grep -c failure || echo 0)
TOTAL=$((SUCC + FAIL))
echo "=== [DONE] Z1 steps_${STEPS} mono_primary: $SUCC/$TOTAL success ($(date)) ==="

kill -9 $SERVER_PID 2>/dev/null || true
