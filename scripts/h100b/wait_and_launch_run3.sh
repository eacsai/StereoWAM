#!/usr/bin/env bash
# Wait until BOTH Run1 (chain) and Run2 (parallel co-located) finish training,
# then launch Run3 on the freed 2 GPUs (Run3 gets both GPUs to itself = full speed).
#
# Why this exists: Run2 was launched in parallel (co-located with Run1) at the
# user's request, OUTSIDE the sequential chain. After Run1 finishes the chain
# tries Run2, sees Run2's ckpts already exist, refuses, and halts -> it will NOT
# auto-launch Run3. This daemon picks up Run3 once the GPUs are actually free.
set -u
cd /mnt/data/wangqiwei/wangqiwei/starVLA
echo "[run3-wait] START $(date -u)"
stable_zero=0
while true; do
  n=$(pgrep -fc 'train_starvla.py' 2>/dev/null || echo 0)
  if [ "$n" -eq 0 ]; then stable_zero=$((stable_zero+1)); else stable_zero=0; fi
  echo "[run3-wait] train_starvla procs=$n stable_zero=$stable_zero $(date -u)"
  # 3 consecutive zero readings (6 min) = both Run1 and Run2 truly done, not a transient gap.
  [ "$stable_zero" -ge 3 ] && break
  sleep 120
done
echo "[run3-wait] GPUs free -> launching Run3 (frozen-head, full 2-GPU) $(date -u)"
bash scripts/h100b/run_pi_qwen0p8_vlmcontrolnet_frozen_head.sh
echo "[run3-wait] Run3 exited rc=$? $(date -u)"
