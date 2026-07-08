#!/usr/bin/env bash
# Start/resume the PLAIN stage-1 (learn_scene_flow) in a tmux ON a800a. $1 = RESUME (0/1).
# Single source of the stage-1 launch command for the keep-alive watcher's auto-resume.
set -uo pipefail
RESUME="${1:-0}"
RUN_ID=qwen0p8_groot_cascade_motion_cambranch_learn_scene_flow_fromscratch_leftprimary
cd /home/wangqiwei/ICLR2026/starVLA
tmux kill-session -t "$RUN_ID" 2>/dev/null || true
tmux new-session -d -s "$RUN_ID" "cd /home/wangqiwei/ICLR2026/starVLA && \
STAGE=learn_scene_flow PRETRAINED_CKPT= CAM_BRANCH=1 TARGET_KEY=flow_gt \
BS=16 MAX_STEPS=30000 SAVE_INTERVAL=1000 WARMUP_STEPS=5000 LOGGING_FREQUENCY=20 \
GPUS=0 PORT=29821 SKIP_CASCADE_SMOKE=1 RESUME=$RESUME RUN_ID=$RUN_ID \
bash scripts/a800/run_sceneflow_cascade.sh >> playground/Checkpoints/$RUN_ID.train.log 2>&1; \
echo TRAIN_EXIT=\$? >> playground/Checkpoints/$RUN_ID.train.log"
sleep 3
tmux has-session -t "$RUN_ID" 2>/dev/null && echo "RELAUNCHED RESUME=$RESUME session=alive" || echo "RELAUNCH_FAILED"
