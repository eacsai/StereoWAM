#!/usr/bin/env bash
# Orchestrator (h100b nohup): wait for the gru_hidden WARM-START training to finish, then
# launch the fromscratch gru_hidden 90k convergence run on the freed GPUs 0,1.
set -uo pipefail
H=/mnt/data/wangqiwei/wangqiwei/starVLA
LOG=$H/playground/Checkpoints/gruhidden_90k_orchestrator.log
echo "[orch-90k] waiting for gruhidden_warmstart training to finish... $(date -u +%H:%M)" >> "$LOG"
for i in $(seq 1 720); do            # up to ~12h
  if ! pgrep -f "train_starvla.*gruhidden_warmstart" >/dev/null 2>&1; then
    echo "[orch-90k] warmstart training finished $(date -u +%H:%M)" >> "$LOG"; break
  fi
  sleep 60
done
sleep 15
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2 >> "$LOG"
cd "$H"
nohup bash scripts/h100b/run_pi_qwen0p8_camrope_controlvla_gruhidden_fromscratch_90k.sh \
  > playground/Checkpoints/gruhidden_fromscratch_90k_0529_launch.log 2>&1 &
echo "[orch-90k] 90k launched pid $! $(date -u +%H:%M)" >> "$LOG"
