#!/usr/bin/env bash
# Start/resume the a800a GPU1 UTONIA STAGE-2 (joint_cascade, warmstart from the Utonia stage-1 flow ckpt).
# $1 = RESUME (0/1). RESUME=0 -> warmstart from the Utonia stage-1 flow ckpt; RESUME=1 -> hot-restart own ckpt.
# Mirror of _resume_joint_stage2.sh but GPU1 + Utonia launcher + Utonia run_id.
set -uo pipefail
RESUME="${1:-0}"
RUN_ID=qwen0p8_groot_cascade_motion_utonia_cambranch_joint_warmstartflow_leftprimary
UTONIA_FLOW_CKPT=/home/wangqiwei/ICLR2026/starVLA/playground/Checkpoints/qwen0p8_groot_cascade_motion_utonia_cambranch_learn_scene_flow_fromscratch_leftprimary/checkpoints/steps_30000_pytorch_model.pt
cd /home/wangqiwei/ICLR2026/starVLA
if [ "$RESUME" = "1" ]; then PRE=; else PRE="$UTONIA_FLOW_CKPT"; fi
tmux kill-session -t "$RUN_ID" 2>/dev/null || true
tmux new-session -d -s "$RUN_ID" "cd /home/wangqiwei/ICLR2026/starVLA && \
STAGE=joint_cascade PRETRAINED_CKPT=$PRE CAM_BRANCH=1 TARGET_KEY=flow_gt \
BS=16 MAX_STEPS=30000 SAVE_INTERVAL=1000 WARMUP_STEPS=5000 LOGGING_FREQUENCY=20 \
GPUS=1 PORT=29827 SKIP_CASCADE_SMOKE=1 RESUME=$RESUME RUN_ID=$RUN_ID \
bash scripts/a800/run_utonia_sceneflow_cascade.sh >> playground/Checkpoints/$RUN_ID.train.log 2>&1; \
echo TRAIN_EXIT=\$? >> playground/Checkpoints/$RUN_ID.train.log"
sleep 3
tmux has-session -t "$RUN_ID" 2>/dev/null && echo "RELAUNCHED RESUME=$RESUME session=alive" || echo "RELAUNCH_FAILED"
