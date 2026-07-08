#!/usr/bin/env bash
# Start/resume the a800a GPU0 STAGE-2 (joint_cascade, warmstart from the plain stage-1 flow ckpt) in a tmux ON a800a.
# $1 = RESUME (0/1).
#   RESUME=0 -> fresh stage-2: warmstart (initialize) from the plain stage-1 flow-pretrained ckpt.
#   RESUME=1 -> preemption recovery: hot-restart from THIS run's own latest ckpt (is_resume), NO re-warmstart.
set -uo pipefail
RESUME="${1:-0}"
RUN_ID=qwen0p8_groot_cascade_motion_cambranch_joint_warmstartflow_leftprimary
PLAIN_FLOW_CKPT=/home/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/qwen0p8_groot_cascade_motion_cambranch_learn_scene_flow_fromscratch_leftprimary/checkpoints/steps_30000_pytorch_model.pt
cd /home/wangqiwei/ICLR2026/starVLA
if [ "$RESUME" = "1" ]; then PRE=; else PRE="$PLAIN_FLOW_CKPT"; fi
tmux kill-session -t "$RUN_ID" 2>/dev/null || true
tmux new-session -d -s "$RUN_ID" "cd /home/wangqiwei/ICLR2026/starVLA && \
STAGE=joint_cascade PRETRAINED_CKPT=$PRE CAM_BRANCH=1 TARGET_KEY=flow_gt FLOW_LAMBDA=0.1 \
BS=16 MAX_STEPS=30000 SAVE_INTERVAL=1000 WARMUP_STEPS=5000 LOGGING_FREQUENCY=20 \
GPUS=0 PORT=29831 SKIP_CASCADE_SMOKE=1 RESUME=$RESUME RUN_ID=$RUN_ID \
bash scripts/a800/run_sceneflow_cascade.sh >> playground/Checkpoints/$RUN_ID.train.log 2>&1; \
echo TRAIN_EXIT=\$? >> playground/Checkpoints/$RUN_ID.train.log"
sleep 3
tmux has-session -t "$RUN_ID" 2>/dev/null && echo "RELAUNCHED RESUME=$RESUME session=alive" || echo "RELAUNCH_FAILED"
