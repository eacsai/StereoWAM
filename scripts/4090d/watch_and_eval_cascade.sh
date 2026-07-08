#!/usr/bin/env bash
# Keep-alive + auto-resume + eval watcher for the A800b scene-flow cascade 30k.
# Survives platform POD RESTARTS on a800b (which SIGKILL the training with no crash marker):
#   - SESSION_GONE + no code-crash marker -> auto-resume (RESUME=1 from latest ckpt, or fresh if none)
#   - real code crash (Traceback/OOM/etc) -> flag CRASHED, do NOT auto-resume (would just re-crash)
#   - step-30000 ckpt -> pull to 4090d + auto-eval 4-suite -> exit
# Runs in a 4090d tmux (stable). Relaunch cap prevents an infinite crash-loop.
set -uo pipefail

RUN_ID="${RUN_ID:-qwen0p8_groot_cascade_motion_cambranch_joint_cascade_fromscratch_leftprimary}"
STEP="${STEP:-30000}"
A800="ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -i $HOME/.ssh/id_a800_push -p 11089 wangqiwei@10.13.32.3"
A800_REPO=/home/wangqiwei/ICLR2026/starVLA
STARVLA=/data/wangqiwei/ICLR2026/starVLA
POLL="${POLL:-180}"
MAX_RELAUNCH="${MAX_RELAUNCH:-12}"

TRAIN_LOG="$A800_REPO/playground/Checkpoints/$RUN_ID.train.log"
CKPT_DIR="$A800_REPO/playground/Checkpoints/$RUN_ID/checkpoints"
CKPT_FINAL="$CKPT_DIR/steps_${STEP}_pytorch_model.pt"
STATUS="$STARVLA/playground/Checkpoints/$RUN_ID.watcher.status"
LOG="$STARVLA/playground/Checkpoints/$RUN_ID.watcher.log"

mark(){ printf '%s\n' "$1" > "$STATUS"; printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$1" >> "$LOG"; }

relaunch_count=0
last_seen_step=0
mark "WATCHING run=$RUN_ID on a800b (keep-alive, poll ${POLL}s, max_relaunch=$MAX_RELAUNCH)"

while true; do
  up=$($A800 "echo UP" 2>/dev/null || true)
  if [ "$up" != "UP" ]; then mark "A800B_UNREACHABLE (transient? retry next poll)"; sleep "$POLL"; continue; fi

  remote_sz=$($A800 "stat -c %s '$CKPT_FINAL' 2>/dev/null" 2>/dev/null | tr -dc 0-9 || true)
  train_exit=$($A800 "grep -oE 'TRAIN_EXIT=[0-9]+' '$TRAIN_LOG' 2>/dev/null | tail -1" 2>/dev/null || true)
  crash=$($A800 "grep -cE 'Traceback|CUDA error|out of memory|RuntimeError|deferred position cache still|AssertionError' '$TRAIN_LOG' 2>/dev/null" 2>/dev/null | tr -dc 0-9 || true)
  session=$($A800 "tmux has-session -t '$RUN_ID' 2>/dev/null && echo ALIVE || echo GONE" 2>/dev/null || echo UNKNOWN)
  cur_step=$($A800 "tail -n 6 '$TRAIN_LOG' 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/${STEP}' | grep -oE '^[0-9]+' | tail -1" 2>/dev/null | tr -dc 0-9 || true)
  [ -n "${cur_step:-}" ] && [ "${cur_step:-0}" -gt "$last_seen_step" ] && { last_seen_step=$cur_step; relaunch_count=0; }  # progress -> reset relaunch budget

  # ---- COMPLETE: final ckpt exists (size-stable) OR clean TRAIN_EXIT=0 with the final ckpt ----
  if [ -n "${remote_sz:-}" ] && [ "${remote_sz:-0}" -gt 1000000 ]; then
    sleep 15; sz2=$($A800 "stat -c %s '$CKPT_FINAL' 2>/dev/null" 2>/dev/null | tr -dc 0-9 || true)
    if [ "$remote_sz" = "$sz2" ]; then
      mark "COMPLETE step=$STEP (ckpt ${remote_sz}B) — pulling a800b->4090d"
      LR="$STARVLA/playground/Checkpoints/$RUN_ID"; mkdir -p "$LR/checkpoints"
      $A800 "cat '$CKPT_FINAL'" > "$LR/checkpoints/steps_${STEP}_pytorch_model.pt"
      $A800 "cat '$A800_REPO/playground/Checkpoints/$RUN_ID/config.full.yaml'" > "$LR/config.full.yaml"
      cp "$LR/config.full.yaml" "$LR/config.yaml"
      $A800 "cat '$A800_REPO/playground/Checkpoints/$RUN_ID/dataset_statistics.json'" > "$LR/dataset_statistics.json"
      local_sz=$(stat -c %s "$LR/checkpoints/steps_${STEP}_pytorch_model.pt" 2>/dev/null | tr -dc 0-9 || echo 0)
      if [ "$local_sz" != "$remote_sz" ]; then mark "PULL_FAIL size local=$local_sz remote=$remote_sz"; exit 2; fi
      EG=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -4 | cut -d, -f1 | tr '\n' ' ' | sed 's/ $//')
      mark "PULLED; launching 4-suite eval on 4090d GPUs [$EG] vk=primary,left_view"
      cd "$STARVLA" && bash scripts/4090d/eval_qwen2p5vl_4suite.sh "$RUN_ID" "$STEP" primary,left_view "$EG" >> "$LOG" 2>&1 || mark "EVAL_ERROR (see watcher.log)"
      mark "EVAL_DONE $(grep -iE 'MEAN' "$LOG" | tail -1)"
      exit 0
    fi
  fi

  # ---- real code crash -> do NOT auto-resume (would re-crash) ----
  if [ -n "${crash:-}" ] && [ "${crash:-0}" != "0" ] && [ "$train_exit" != "TRAIN_EXIT=0" ]; then
    mark "CRASHED (code error) — $($A800 "grep -E 'Traceback|Error|out of memory' '$TRAIN_LOG' 2>/dev/null | tail -3" 2>/dev/null)"
    exit 0
  fi

  # ---- SESSION GONE (pod restart / SIGKILL, no crash marker) -> AUTO-RESUME ----
  if [ "$session" = "GONE" ] && [ "$train_exit" != "TRAIN_EXIT=0" ]; then
    relaunch_count=$((relaunch_count + 1))
    if [ "$relaunch_count" -gt "$MAX_RELAUNCH" ]; then
      mark "GAVE_UP: $MAX_RELAUNCH relaunches without lasting progress (last_step=$last_seen_step) — likely persistent preemption or a fast-death bug. Manual check needed."
      exit 0
    fi
    any_ckpt=$($A800 "ls '$CKPT_DIR'/steps_*_pytorch_model.pt 2>/dev/null | head -1" 2>/dev/null || true)
    if [ -n "$any_ckpt" ]; then
      $A800 "cd '$A800_REPO' && bash scripts/a800/_resume_cascade.sh 1" >> "$LOG" 2>&1 || true
      mark "RESUMED (#$relaunch_count) from last ckpt (last_step=$last_seen_step)"
    else
      $A800 "cd '$A800_REPO' && bash scripts/a800/_resume_cascade.sh 0" >> "$LOG" 2>&1 || true
      mark "RELAUNCHED_FRESH (#$relaunch_count, no ckpt yet)"
    fi
    sleep 120
    continue
  fi

  mark "RUNNING ${cur_step:-<pre-train>}/${STEP} (relaunches=$relaunch_count)"
  sleep "$POLL"
done
