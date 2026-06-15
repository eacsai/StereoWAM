#!/usr/bin/env bash
# Wait for the existing queuewatch (queue_remaining: pending #8 warmstartB) to finish, THEN run the
# method10 utonia GPU-wait queue. Avoids two concurrent GPU waiters racing for the same freed card.
set -uo pipefail
REPO=/data/wangqiwei/ICLR2026/starVLA
echo "[m10_after_queue $(date -u +%FT%TZ)] waiting for queuewatch tmux to finish (#8 warmstartB launch)..."
while tmux has-session -t queuewatch 2>/dev/null; do sleep 60; done
echo "[m10_after_queue $(date -u +%FT%TZ)] queuewatch gone -> starting utonia method10 queue (perpatch -> resampler)"
bash "$REPO/scripts/4090d/wait_gpu_launch_method10.sh"
echo "[m10_after_queue $(date -u +%FT%TZ)] method10 queue finished"
