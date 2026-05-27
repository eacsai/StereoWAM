#!/usr/bin/env bash
# Sweep eval of 3B ckpts on GPU 0 — parallel with epipolar sweep on GPU 2.
set -euo pipefail

UPSTREAM_DIR=/data/wangqiwei/ICLR2026/starVLA
RUN_ID=upstream_groot_qwen2p5vl3b_libero_goal_h100b_0526
PORT=${PORT:-6698}
GPU=${GPU:-0}
TASK_SUITE=${TASK_SUITE:-libero_goal}
NUM_TRIALS=${NUM_TRIALS:-10}

LIBERO_VENV=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python
SERVER_VENV=${UPSTREAM_DIR}/.venv/bin/python

for STEP in 10000 15000 20000 25000 30000; do
  CKPT=${UPSTREAM_DIR}/playground/Checkpoints/${RUN_ID}/checkpoints/steps_${STEP}_pytorch_model.pt
  EVAL_DIR=${UPSTREAM_DIR}/playground/Checkpoints/${RUN_ID}/eval_logs/step${STEP}_primary_wrist

  [ ! -f "$CKPT" ] && { echo "skip step $STEP: ckpt missing"; continue; }
  [ -f "$EVAL_DIR/done.txt" ] && { echo "skip step $STEP: already done"; continue; }
  mkdir -p "$EVAL_DIR/videos"

  echo "============================================================"
  echo "=== 3B sweep: step $STEP  GPU=$GPU port=$PORT ==="
  echo "============================================================"

  pkill -9 -f "server_policy.py.*--port $PORT" || true
  sleep 3

  cd "$UPSTREAM_DIR"
  CUDA_VISIBLE_DEVICES=$GPU nohup "$SERVER_VENV" deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port $PORT --use_bf16 \
    > "$EVAL_DIR/server.log" 2>&1 &
  SERVER_PID=$!

  for i in $(seq 1 120); do
    grep -q "listening" "$EVAL_DIR/server.log" 2>/dev/null && { echo "[server ready]"; break; }
    ps -p $SERVER_PID > /dev/null 2>&1 || { echo "[server died]"; tail -50 "$EVAL_DIR/server.log"; break; }
    sleep 2
  done

  export LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO
  export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
  export PYTHONPATH=${LIBERO_HOME}:$UPSTREAM_DIR:${PYTHONPATH:-}
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl

  $LIBERO_VENV examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port $PORT \
    --args.task-suite-name $TASK_SUITE --args.num-trials-per-task $NUM_TRIALS --args.max-tasks -1 \
    --args.video-out-path "$EVAL_DIR/videos" \
    > "$EVAL_DIR/client.log" 2>&1 || true

  SUCC=$(ls "$EVAL_DIR/videos" 2>/dev/null | grep -c success || true)
  FAIL=$(ls "$EVAL_DIR/videos" 2>/dev/null | grep -c failure || true)
  echo "=== [step $STEP] $SUCC/$((SUCC+FAIL)) success ==="
  echo "$SUCC/$((SUCC+FAIL))" > "$EVAL_DIR/done.txt"
  pkill -9 -f "server_policy.py.*--port $PORT" || true
done
echo "[3B SWEEP DONE]"
