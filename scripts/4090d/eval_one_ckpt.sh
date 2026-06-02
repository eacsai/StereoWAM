#!/usr/bin/env bash
# Generic single-ckpt LIBERO eval lane: starVLA policy server (1 GPU) + LIBERO client (CPU + EGL mujoco).
# Replaces the hardcoded per-run launchers (eval_mono_step / eval_y_stereo / eval_ckpt_sweep) with one
# parameterized entrypoint, so parallel multi-GPU sweeps just launch N copies on different GPU/port pairs.
#
# Usage: eval_one_ckpt.sh <CKPT_PATH> <GPU> <PORT> <EVAL_DIR> [TASK_SUITE] [VIDEO_KEYS]
#   CKPT_PATH  abs path to steps_<N>_pytorch_model.pt
#   GPU        CUDA device index dedicated to this lane's server
#   PORT       policy server port (unique per lane)
#   EVAL_DIR   output dir for server.log / client.log / videos
#   TASK_SUITE libero suite name           (default: libero_goal)
#   VIDEO_KEYS comma-separated camera keys  (default: primary,right_view  -- STEREO models, NO wrist)
#              mono ckpts: "primary" ; stereo ckpts: "primary,right_view". Never "primary,wrist" here.
set -euo pipefail

CKPT="$1"; GPU="$2"; PORT="$3"; EVAL_DIR="$4"; TASK_SUITE="${5:-libero_goal}"; VIDEO_KEYS="${6:-primary,right_view}"
STARVLA_DIR=/data/wangqiwei/ICLR2026/starVLA
LIBERO_VENV=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python
LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO

[ -f "$CKPT" ] || { echo "[FATAL] ckpt not found: $CKPT"; exit 1; }
mkdir -p "$EVAL_DIR/videos"

echo "=== eval $CKPT  GPU=$GPU port=$PORT suite=$TASK_SUITE video_keys=$VIDEO_KEYS  $(date) ==="

pkill -9 -f "server_policy.py.*--port $PORT" 2>/dev/null || true
sleep 3

cd "$STARVLA_DIR"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port "$PORT" --use_bf16 \
  > "$EVAL_DIR/server.log" 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 90); do
  if grep -q "listening" "$EVAL_DIR/server.log" 2>/dev/null; then
    echo "  [server ready iter $i pid $SERVER_PID]"; READY=1; break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "  [FATAL] server died before ready, see $EVAL_DIR/server.log"; exit 2
  fi
  sleep 2
done
[ "$READY" = 1 ] || { echo "  [FATAL] server not ready in time"; kill -9 $SERVER_PID 2>/dev/null || true; exit 3; }

export LIBERO_HOME LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

$LIBERO_VENV examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port "$PORT" \
  --args.task-suite-name "$TASK_SUITE" --args.num-trials-per-task 10 --args.max-tasks -1 \
  --args.video-keys "$VIDEO_KEYS" \
  --args.video-out-path "$EVAL_DIR/videos" \
  > "$EVAL_DIR/client.log" 2>&1 || true

echo "  [RESULT] $(grep 'Total success rate' "$EVAL_DIR/client.log" | tail -1)  ($(date))"
kill -9 $SERVER_PID 2>/dev/null || true
sleep 3
echo "=== lane done: $CKPT ==="
