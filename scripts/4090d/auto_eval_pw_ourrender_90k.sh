#!/usr/bin/env bash
set -euo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
cat >&2 <<'MSG'
[FATAL] This legacy remote-checkpoint auto-eval daemon is retired.
Use a current source-specific watcher/puller (for example scripts/4090d/auto_eval_30k_a800.sh)
to place ckpt/config/stats under playground/Checkpoints/<RUN_ID>/ on 4090d, then run:
  scripts/4090d/eval_qwen2p5vl_4suite.sh <RUN_ID> <STEP> [VIDEO_KEYS] [GPUS]
MSG
exit 64
