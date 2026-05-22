#!/usr/bin/env bash
# Generic mono-primary single-image eval launcher.
# Same model_server + LIBERO eval client recipe as eval_z1_step10000_mono.sh,
# but RUN_ID + STEPS + GPU + PORT all parametrized via env vars so we can:
#   - eval the OTHER 2 mono trainings (Z self-rendered, future suites)
#   - sweep Z1 ckpts (15k/20k/25k/30k) with the same script
set -euo pipefail

RUN_ID=${RUN_ID:?must set RUN_ID (e.g. goal_only_30k_mono_primary_official_0520 or goal_mono_replay_30k_qwenpi_0520)}
STEPS=${STEPS:?must set STEPS (e.g. 15000)}
PORT=${PORT:-6695}
GPU=${GPU:-6}
TASK_SUITE=${TASK_SUITE:-libero_goal}
NUM_TRIALS=${NUM_TRIALS:-10}
GRIPPER_CONVENTION=${GRIPPER_CONVENTION:-openvla}

STARVLA_DIR=/data/wangqiwei/ICLR2026/starVLA
LIBERO_VENV=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python
CKPT_DIR=$STARVLA_DIR/playground/Checkpoints/$RUN_ID/checkpoints
RUN_ROOT=$STARVLA_DIR/playground/Checkpoints/$RUN_ID

CKPT=$CKPT_DIR/steps_${STEPS}_pytorch_model.pt
EVAL_DIR=$RUN_ROOT/eval_logs/step${STEPS}_mono_primary

[ ! -f "$CKPT" ] && { echo "ABORT: $CKPT not found"; exit 1; }
mkdir -p $EVAL_DIR/videos

echo "=== eval mono $RUN_ID step $STEPS -> $EVAL_DIR ($(date)) ==="
echo "    GPU=$GPU port=$PORT video_keys=primary"

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

echo "  [client] launching"
export LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:$STARVLA_DIR:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
$LIBERO_VENV examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path $CKPT --args.host 127.0.0.1 --args.port $PORT \
  --args.task-suite-name $TASK_SUITE --args.num-trials-per-task $NUM_TRIALS --args.max-tasks -1 \
  --args.video-out-path $EVAL_DIR/videos \
  --args.video-keys primary --args.gripper-convention $GRIPPER_CONVENTION \
  > $EVAL_DIR/client.log 2>&1 || true

SUCC=$(ls $EVAL_DIR/videos 2>/dev/null | grep -c success || true)
FAIL=$(ls $EVAL_DIR/videos 2>/dev/null | grep -c failure || true)
TOTAL=$((SUCC + FAIL))
echo "=== [DONE] $RUN_ID step $STEPS mono: $SUCC/$TOTAL success ($(date)) ==="

kill -9 $SERVER_PID 2>/dev/null || true
