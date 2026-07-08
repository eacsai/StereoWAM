#!/usr/bin/env bash
# Start/resume the scene-flow cascade 30k in a tmux ON THIS BOX (a800b). Single source of the
# launch command so the 4090d keep-alive watcher can auto-resume after a platform pod restart.
#   $1 = RESUME (0 = fresh, 1 = resume from latest ckpt). Default 0.
set -uo pipefail
RESUME="${1:-0}"
RUN_ID=qwen0p8_groot_cascade_motion_cambranch_joint_cascade_fromscratch_leftprimary
cd /home/wangqiwei/ICLR2026/starVLA
tmux kill-session -t "$RUN_ID" 2>/dev/null || true
tmux new-session -d -s "$RUN_ID" "cd /home/wangqiwei/ICLR2026/starVLA && \
STAGE=joint_cascade PRETRAINED_CKPT= CAM_BRANCH=1 TARGET_KEY=flow_gt \
BS=16 MAX_STEPS=30000 SAVE_INTERVAL=1000 WARMUP_STEPS=5000 LOGGING_FREQUENCY=20 \
GPUS=0 PORT=29817 SKIP_CASCADE_SMOKE=1 RESUME=$RESUME RUN_ID=$RUN_ID \
bash scripts/a800/run_sceneflow_cascade.sh >> playground/Checkpoints/$RUN_ID.train.log 2>&1; \
echo TRAIN_EXIT=\$? >> playground/Checkpoints/$RUN_ID.train.log"
sleep 3
tmux has-session -t "$RUN_ID" 2>/dev/null && echo "RELAUNCHED RESUME=$RESUME session=alive" || echo "RELAUNCH_FAILED RESUME=$RESUME"
