#!/usr/bin/env bash
# Sweep eval the remaining unEvaled ckpts of goal_only_30k_qwenpi_0519 on libero_goal.
# Server: GPU 3 + port 6694. Client: LIBERO venv (CPU + EGL mujoco). Serial, one ckpt at a time.
set -euo pipefail

CKPT_DIR=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/goal_only_30k_qwenpi_0519/checkpoints
RUN_ROOT=/data/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/goal_only_30k_qwenpi_0519
STARVLA_DIR=/data/wangqiwei/ICLR2026/starVLA
LIBERO_VENV=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python
PORT=6694
GPU=3
TASK_SUITE=libero_goal

for STEPS in 25000 20000 15000 10000; do
  CKPT=$CKPT_DIR/steps_${STEPS}_pytorch_model.pt
  EVAL_DIR=$RUN_ROOT/eval_logs/step${STEPS}
  echo "=== eval steps_${STEPS} -> $EVAL_DIR ($(date)) ==="

  [ ! -f "$CKPT" ] && { echo "SKIP: $CKPT not found"; continue; }
  if [ -d "$EVAL_DIR/videos" ] && [ "$(ls $EVAL_DIR/videos 2>/dev/null | wc -l)" -ge 100 ]; then
    echo "SKIP: $EVAL_DIR already has 100+ episodes"; continue
  fi
  mkdir -p $EVAL_DIR/videos

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
      echo "  [server ready at iter $i, PID=$SERVER_PID]"; break
    fi
    sleep 2
  done

  echo "  [client] launching"
  export LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO
  export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
  export PYTHONPATH=${LIBERO_HOME}:$STARVLA_DIR:${PYTHONPATH:-}
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  $LIBERO_VENV examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path $CKPT --args.host 127.0.0.1 --args.port $PORT \
    --args.task-suite-name $TASK_SUITE --args.num-trials-per-task 10 --args.max-tasks -1 \
    --args.video-out-path $EVAL_DIR/videos \
    > $EVAL_DIR/client.log 2>&1 || true

  SUCC=$(ls $EVAL_DIR/videos 2>/dev/null | grep -c success || echo 0)
  FAIL=$(ls $EVAL_DIR/videos 2>/dev/null | grep -c failure || echo 0)
  TOTAL=$((SUCC + FAIL))
  echo "  [done] steps_${STEPS}: $SUCC/$TOTAL success ($(date))"

  kill -9 $SERVER_PID 2>/dev/null || true
  sleep 5
done

echo "=== SWEEP ALL DONE $(date) ==="
