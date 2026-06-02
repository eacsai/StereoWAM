#!/usr/bin/env bash
# Rerun Run2 + Run3 after the 2026-06-01 crash. Run1 (head-only) already completed (eval 0.88/0.91/0.92).
#
# Crash root cause: at per-GPU micro-batch 48 the frozen FoundationStereo forward overflows cuDNN's
# grid_sample in the cost volume -> RuntimeError: CUDNN_STATUS_NOT_SUPPORTED (FFS geometry, step 0).
# Run1 survived BS48 only because head-only has no FFS forward. The earlier "co-location" diagnosis was
# wrong -- sequential runs crash too because they are also BS48.
#
# Fix (proven; controlvla_gruhidden FFS run used micro-batch 24): per-GPU micro-batch 24 + GA2
# (deepspeed_zero2_ga2.yaml) = eff_batch 24 x 2GPU x 2 = 96, identical eff_batch to Run1 so the
# ablation stays fair. Set via env override; the launchers are unchanged.
#
# Reuses the _0601 run_ids (RUN_ID override) so identity is stable across the crash+rerun and the
# 4090d eval orchestrator keys onto the same dirs. Run2 reuses its empty crashed dir (resume guard
# sees 0 ckpts via nullglob and proceeds). Halt-on-fail: if Run2 fails again, Run3 is not launched.
set -u
cd /mnt/data/wangqiwei/wangqiwei/starVLA

export BS=24
export DS_CONFIG=starVLA/config/deepseeds/deepspeed_zero2_ga2.yaml   # GA=2; launcher forces --num_processes 2

echo "[chain23] START $(date -u)  BS=$BS  DS_CONFIG=$DS_CONFIG  -> eff_batch = 24 x 2GPU x GA2 = 96"
RUNS=(
  "run2_trainhead:scripts/h100b/run_pi_qwen0p8_vlmcontrolnet_train_head.sh:pi_qwen0p8_vlmcontrolnet_ffs_trainhead_warmstart_0601"
  "run3_frozenhead:scripts/h100b/run_pi_qwen0p8_vlmcontrolnet_frozen_head.sh:pi_qwen0p8_vlmcontrolnet_ffs_frozenhead_warmstart_0601"
)
for spec in "${RUNS[@]}"; do
  name=${spec%%:*}; rest=${spec#*:}; sh=${rest%%:*}; rid=${rest##*:}
  echo "[chain23] >>> $name ($sh) run_id=$rid at $(date -u)"
  RUN_ID="$rid" bash "$sh"
  rc=$?
  echo "[chain23] <<< $name rc=$rc at $(date -u)"
  if [ "$rc" -ne 0 ]; then
    echo "[chain23] HALT: $name failed (rc=$rc); not launching the rest. Investigate, then relaunch remaining."
    exit "$rc"
  fi
done
echo "[chain23] ALL DONE $(date -u)"
