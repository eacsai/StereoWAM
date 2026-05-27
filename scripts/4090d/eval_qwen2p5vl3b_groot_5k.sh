#!/usr/bin/env bash
# Eval upstream-Qwen2.5-VL-3B + GR00T ckpt (trained on h100b) on 4090d.
# Compare to 4B counterpart (which was 0/100 NaN — Issue 171). 3B should give >0% if weights healthy.
set -euo pipefail

UPSTREAM_DIR=/data/wangqiwei/ICLR2026/starVLA
RUN_ID=upstream_groot_qwen2p5vl3b_libero_goal_h100b_0526
STEPS=${STEPS:-5000}
PORT=${PORT:-6698}
GPU=${GPU:-0}
TASK_SUITE=${TASK_SUITE:-libero_goal}
NUM_TRIALS=${NUM_TRIALS:-10}

LIBERO_VENV=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python
SERVER_VENV=${UPSTREAM_DIR}/.venv/bin/python

CKPT=${UPSTREAM_DIR}/playground/Checkpoints/${RUN_ID}/checkpoints/steps_${STEPS}_pytorch_model.pt
EVAL_DIR=${UPSTREAM_DIR}/playground/Checkpoints/${RUN_ID}/eval_logs/step${STEPS}_primary_wrist

[ ! -f "$CKPT" ] && { echo "ABORT: $CKPT not found"; exit 1; }
mkdir -p "$EVAL_DIR/videos"

echo "=== eval $RUN_ID step $STEPS  GPU=$GPU port=$PORT ==="

pkill -9 -f "server_policy.py.*--port $PORT" || true
sleep 3

cd "$UPSTREAM_DIR"
echo "[server] launching..."
CUDA_VISIBLE_DEVICES=$GPU nohup "$SERVER_VENV" deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port $PORT --use_bf16 \
  > "$EVAL_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "[server PID=$SERVER_PID]"

for i in $(seq 1 120); do
  if grep -q "listening" "$EVAL_DIR/server.log" 2>/dev/null; then
    echo "[server ready at iter $i]"; break
  fi
  if ! ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "[server died — tail of server.log]"; tail -50 "$EVAL_DIR/server.log"; exit 1
  fi
  sleep 2
done

echo "[client] launching (NUM_TRIALS=$NUM_TRIALS per task, $TASK_SUITE)..."
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
TOTAL=$((SUCC + FAIL))
echo "=== [DONE] $RUN_ID step $STEPS  $SUCC/$TOTAL success ==="
pkill -9 -f "server_policy.py.*--port $PORT" || true
