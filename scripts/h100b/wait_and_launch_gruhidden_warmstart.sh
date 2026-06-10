#!/usr/bin/env bash
# Orchestrator (h100b, nohup): wait for the gru_hidden FROMSCRATCH training to finish,
# then launch the gru_hidden WARM-START run on the freed GPUs 0,1. Autonomous so it
# survives the operator's session. Pairs with the fromscratch run for warm-start-vs-scratch.
set -uo pipefail
H=/mnt/data/wangqiwei/wangqiwei/starVLA
LOG=$H/playground/Checkpoints/gruhidden_warmstart_orchestrator.log
echo "[orchestrator] waiting for gruhidden_fromscratch training to finish... $(date -u +%H:%M)" >> "$LOG"
for i in $(seq 1 300); do            # up to ~5h
  if ! pgrep -f gruhidden_fromscratch >/dev/null 2>&1; then
    echo "[orchestrator] fromscratch training finished $(date -u +%H:%M)" >> "$LOG"; break
  fi
  sleep 60
done
sleep 15
echo "[orchestrator] GPU before launch:" >> "$LOG"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2 >> "$LOG"
cd "$H"
nohup bash scripts/h100b/run_pi_qwen0p8_camrope_controlvla_gruhidden_warmstart.sh \
  > playground/Checkpoints/gruhidden_warmstart_0529_launch.log 2>&1 &
echo "[orchestrator] warmstart launched pid $! $(date -u +%H:%M)" >> "$LOG"
