#!/usr/bin/env bash
# Sequentially run the 3 VLM-ControlNet-FFS ablation runs on h100b.
# Each run uses BOTH h100b GPUs (eff_batch 96 = BS24 x 2GPU x GA2), so they
# cannot overlap -> run them one after another. Run N starts only if Run N-1
# exited 0 (a startup-time failure in Run1 would also break Run2/3, so halt
# and let a human investigate rather than burn GPU hours).
#
#   Run1 = head-only control (QwenPI, freeze VLM, no FFS)        -> isolates head adaptation
#   Run2 = head + ControlNet (freeze VLM)                        -> FFS effect w/ trainable head
#   Run3 = pure ControlNet   (freeze VLM + action head)          -> FFS effect on frozen 0.94 policy
set -u
cd /mnt/data/wangqiwei/wangqiwei/starVLA
echo "[chain] START $(date -u)"
RUNS=(
  "run1_headonly:scripts/h100b/run_pi_qwen0p8_vlmcontrolnet_headonly_control.sh"
  "run2_trainhead:scripts/h100b/run_pi_qwen0p8_vlmcontrolnet_train_head.sh"
  "run3_frozenhead:scripts/h100b/run_pi_qwen0p8_vlmcontrolnet_frozen_head.sh"
)
for spec in "${RUNS[@]}"; do
  name=${spec%%:*}; sh=${spec#*:}
  echo "[chain] >>> launching $name ($sh) at $(date -u)"
  bash "$sh"
  rc=$?
  echo "[chain] <<< $name exited rc=$rc at $(date -u)"
  if [ "$rc" -ne 0 ]; then
    echo "[chain] HALT: $name failed (rc=$rc); not launching the rest. Investigate, then relaunch remaining."
    exit "$rc"
  fi
done
echo "[chain] ALL 3 RUNS COMPLETE $(date -u)"
