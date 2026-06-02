#!/usr/bin/env bash
# Auto-eval the 3 VLM-ControlNet-FFS ablation runs as their ckpts land on h100b.
#   Run1 head-only control   (QwenPI, freeze VLM, train head, no FFS)
#   Run2 head + ControlNet   (QwenPIVLMControlNetFFS, freeze VLM, train head + branch)
#   Run3 pure ControlNet     (QwenPIVLMControlNetFFS, freeze VLM + action head, train branch only)
#
# For each run: resolve the run dir by PREFIX on h100b (the run_id date suffix is set by the
# launcher's $(date +%m%d) at launch time, so don't hardcode it), then for each 10k/20k/30k ckpt
# wait-until-fully-written -> transfer h100b->4090d -> eval libero_goal primary,right_view (gripper
# openvla, NO wrist) -> log SR. Ends with a 3-way comparison vs the cam_rope 0.94 warm-start baseline.
#
# Follows the scripts/4090d/auto_eval_gruhidden_run.sh pattern but is self-contained, adding:
# prefix-resolution (uncertain date suffix), a generous ckpt-wait that covers far-future runs
# (Run3 starts ~10h after Run1), and the combined decomposition summary.
#
# Runs ON 4090d (the eval servers use 4090d GPUs; ckpts are pulled from h100b over ssh). nohup it.
# Idempotent: re-running skips any step already recorded in a run's auto_eval_results.txt.
# EVAL ONLY -- never launches training. Single-ckpt eval is correct here: the trainer saves the
# full model (frozen trunk + ffs + control branch + action head), and the framework's load_state_dict
# only copy-inits the branch from trunk when branch keys are ABSENT (warm-start), so a trained ckpt's
# branch weights are preserved at eval.
set -uo pipefail

STARVLA=/data/wangqiwei/ICLR2026/starVLA
EVAL=$STARVLA/scripts/4090d/eval_one_ckpt.sh
R="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 root@10.112.2.93"
H_BASE=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints
DST_BASE=$STARVLA/playground/Checkpoints
STEPS=${STEPS:-"10000 20000 30000"}
BASELINE=${BASELINE:-0.94}            # cam_rope 0.94 warm-start reference for the decomposition
RESOLVE_WAIT_MIN=${RESOLVE_WAIT_MIN:-1200}   # how long to wait for a run dir to appear (~20h, covers Run3)
CKPT_WAIT_ITERS=${CKPT_WAIT_ITERS:-3000}     # 20s/iter -> ~16.7h budget per ckpt (covers far-future runs)
SUMMARY=$DST_BASE/vlmcontrolnet_3way_comparison.txt

# label | h100b run-dir prefix | gpu (4090d) | server port  -- GPUs 3/4/6 confirmed free on 4090d
RUNS=(
  "run1_headonly|pi_qwen0p8_camrope_frozenvlm_headonly_warmstart_|3|6720"
  "run2_trainhead|pi_qwen0p8_vlmcontrolnet_ffs_trainhead_warmstart_|4|6721"
  "run3_frozenhead|pi_qwen0p8_vlmcontrolnet_ffs_frozenhead_warmstart_|6|6722"
)

# Resolve a unique run_id on h100b by prefix (newest dir whose config.full.yaml exists). Echoes "" if none.
resolve_run_id() {
  local prefix=$1
  $R "ls -dt ${H_BASE}/${prefix}*/ 2>/dev/null" 2>/dev/null \
    | while read -r d; do $R "test -f ${d}config.full.yaml" 2>/dev/null && { basename "$d"; break; }; done \
    | head -1
}

eval_one_run() {
  local label=$1 prefix=$2 gpu=$3 port=$4
  local run_id="" i
  for i in $(seq 1 "$RESOLVE_WAIT_MIN"); do
    run_id=$(resolve_run_id "$prefix")
    [ -n "$run_id" ] && break
    sleep 60
  done
  [ -z "$run_id" ] && { echo "[$label] TIMEOUT resolving run dir for prefix ${prefix}* after ${RESOLVE_WAIT_MIN}min"; return 1; }

  local H=$H_BASE/$run_id DST=$DST_BASE/$run_id RESULTS=$DST_BASE/$run_id/auto_eval_results.txt
  mkdir -p "$DST/checkpoints"
  echo "[$label] resolved run_id=$run_id gpu=$gpu port=$port $(date -u +%H:%M)" | tee -a "$RESULTS"

  # set up 4090d eval config (idempotent): full resolved config + dataset stats + FFS path rewrite
  if [ ! -f "$DST/config.yaml" ]; then
    $R "cat $H/config.full.yaml" > "$DST/config.yaml"
    $R "cat $H/dataset_statistics.json" > "$DST/dataset_statistics.json"
    sed -i "s#/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo#/data/wangqiwei/ICLR2026/Fast-FoundationStereo#g" "$DST/config.yaml"
    echo "[$label] eval config set up $(date -u +%H:%M)" >> "$RESULTS"
  fi

  local STEP CK EVDIR sz prev ok lsz SR
  for STEP in $STEPS; do
    grep -q "step${STEP}:" "$RESULTS" 2>/dev/null && { echo "[$label] step$STEP already done"; continue; }
    CK=$DST/checkpoints/steps_${STEP}_pytorch_model.pt
    EVDIR=$DST/eval_logs_rightview/step${STEP}
    prev=-1; ok=0
    for i in $(seq 1 "$CKPT_WAIT_ITERS"); do
      sz=$($R "stat -c %s $H/checkpoints/steps_${STEP}_pytorch_model.pt 2>/dev/null" 2>/dev/null || echo 0); sz=${sz:-0}
      if [ "$sz" -gt 1000000 ] && [ "$sz" = "$prev" ]; then ok=1; break; fi
      prev=$sz; sleep 20
    done
    [ "$ok" != 1 ] && { echo "step${STEP}: TIMEOUT_WAIT_CKPT ($(date -u +%H:%M))" >> "$RESULTS"; continue; }
    echo "[$label] transfer step$STEP ($sz bytes) $(date -u +%H:%M)"
    $R "cat $H/checkpoints/steps_${STEP}_pytorch_model.pt" > "$CK"
    lsz=$(stat -c %s "$CK" 2>/dev/null || echo 0)
    [ "$lsz" != "$sz" ] && { echo "step${STEP}: TRANSFER_MISMATCH h100b=$sz 4090d=$lsz" >> "$RESULTS"; continue; }
    echo "[$label] eval step$STEP gpu=$gpu port=$port $(date -u +%H:%M)"
    bash "$EVAL" "$CK" "$gpu" "$port" "$EVDIR" libero_goal primary,right_view > "$DST/auto_eval_step${STEP}.log" 2>&1
    SR=$(grep "Total success rate" "$EVDIR/client.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
    echo "step${STEP}: SR=${SR:-FAILED} ($(date -u +%H:%M))" >> "$RESULTS"
    echo "[$label] step$STEP SR=${SR:-FAILED}"
  done
}

# Launch the 3 watchers in parallel (distinct GPU/port). The runs train sequentially, so in practice
# only one watcher is actively evaling at a time; the others sit in their ckpt-wait poll.
for spec in "${RUNS[@]}"; do
  IFS='|' read -r label prefix gpu port <<< "$spec"
  eval_one_run "$label" "$prefix" "$gpu" "$port" &
done
wait

# 3-way comparison summary (use the 30k row for the decomposition).
{
  echo "=== VLM-ControlNet-FFS 3-way comparison  ($(date -u)) ==="
  echo "baseline (cam_rope 0.94 warm-start) = $BASELINE  | suite=libero_goal  video=primary,right_view"
  for spec in "${RUNS[@]}"; do
    IFS='|' read -r label prefix gpu port <<< "$spec"
    run_id=$(resolve_run_id "$prefix"); run_id=${run_id:-"<unresolved:${prefix}*>"}
    echo "--- $label ($run_id) ---"
    grep -E "step[0-9]+: SR" "$DST_BASE/$run_id/auto_eval_results.txt" 2>/dev/null || echo "  (no results)"
  done
  echo
  echo "Decomposition (30k SR):"
  echo "  Run3 - baseline  = pure FFS-ControlNet effect on the frozen 0.94 policy"
  echo "  Run2 - Run1      = FFS effect when the action head can also adapt"
  echo "  Run1 - baseline  = pure head re-adaptation (no FFS) under a frozen VLM"
} | tee "$SUMMARY"
echo "[auto-eval] ALL DONE $(date -u) -> $SUMMARY"
