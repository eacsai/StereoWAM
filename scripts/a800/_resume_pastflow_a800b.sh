#!/usr/bin/env bash
# Start/resume the a800b PAST-FLOW run (joint_cascade from-scratch + PAST_FLOW=1) in a tmux ON a800b.
# $1 = RESUME (0/1). Directly comparable to the joint-from-scratch 0.85 baseline (same protocol,
# only addition = past-flow ControlNet). a800b single A800-80GB, GPU0.
set -uo pipefail
RESUME="${1:-0}"
RUN_ID=qwen0p8_groot_cascade_motion_cambranch_pastflow_joint_fromscratch_leftprimary
cd /home/wangqiwei/ICLR2026/starVLA
tmux kill-session -t "$RUN_ID" 2>/dev/null || true
tmux new-session -d -s "$RUN_ID" "cd /home/wangqiwei/ICLR2026/starVLA && \
REPO_DIR=/home/wangqiwei/ICLR2026/starVLA STAGE=joint_cascade PRETRAINED_CKPT= \
PAST_FLOW=1 PAST_FLOW_K=2 PAST_FLOW_DELTA=1 PAST_FLOW_DROPOUT=0.3 PAST_FLOW_NOISE=0.0 \
CAM_BRANCH=1 TARGET_KEY=flow_gt BS=16 MAX_STEPS=30000 SAVE_INTERVAL=1000 WARMUP_STEPS=5000 \
LOGGING_FREQUENCY=20 GPUS=0 PORT=29835 SKIP_CASCADE_SMOKE=1 RESUME=$RESUME RUN_ID=$RUN_ID \
bash scripts/a800/run_sceneflow_cascade.sh >> playground/Checkpoints/$RUN_ID.train.log 2>&1; \
echo TRAIN_EXIT=\$? >> playground/Checkpoints/$RUN_ID.train.log"
sleep 3
tmux has-session -t "$RUN_ID" 2>/dev/null && echo "RELAUNCHED RESUME=$RESUME session=alive" || echo "RELAUNCH_FAILED"
