#!/usr/bin/env bash
# Keep-alive + auto-resume watcher for the a800a GPU1 UTONIA stage-1 (learn_scene_flow, flow_only).
# NO eval (flow_only has no policy). On completion -> mark STAGE1_DONE (await manual stage-2 decision).
# SESSION_GONE without code-crash -> auto-resume (RESUME=1 from latest ckpt, else fresh). Runs in a 4090d tmux.
# Mirror of watch_stage1_plain.sh but GPU1 Utonia run_id + Utonia resume helper.
set -uo pipefail

RUN_ID=qwen0p8_groot_cascade_motion_utonia_cambranch_learn_scene_flow_fromscratch_leftprimary
STEP=30000
A800A="ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -i $HOME/.ssh/id_a800_push -p 30947 wangqiwei@10.13.32.4"
A800_REPO=/home/wangqiwei/ICLR2026/starVLA
STARVLA=/data/wangqiwei/ICLR2026/starVLA
POLL=180
MAX_RELAUNCH=12

TRAIN_LOG="$A800_REPO/playground/Checkpoints/$RUN_ID.train.log"
CKPT_DIR="$A800_REPO/playground/Checkpoints/$RUN_ID/checkpoints"
CKPT_FINAL="$CKPT_DIR/steps_${STEP}_pytorch_model.pt"
STATUS="$STARVLA/playground/Checkpoints/$RUN_ID.watcher.status"
LOG="$STARVLA/playground/Checkpoints/$RUN_ID.watcher.log"

mark(){ printf '%s\n' "$1" > "$STATUS"; printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$1" >> "$LOG"; }

relaunch_count=0
last_seen_step=0
mark "WATCHING Utonia stage-1 on a800a GPU1 (keep-alive, no-eval, poll ${POLL}s)"

while true; do
  up=$($A800A "echo UP" 2>/dev/null || true)
  if [ "$up" != "UP" ]; then mark "A800A_UNREACHABLE (transient? retry)"; sleep "$POLL"; continue; fi

  remote_sz=$($A800A "stat -c %s '$CKPT_FINAL' 2>/dev/null" 2>/dev/null | tr -dc 0-9 || true)
  train_exit=$($A800A "grep -oE 'TRAIN_EXIT=[0-9]+' '$TRAIN_LOG' 2>/dev/null | tail -1" 2>/dev/null || true)
  crash=$($A800A "grep -cE 'Traceback|CUDA error|out of memory|RuntimeError|deferred position cache still|AssertionError' '$TRAIN_LOG' 2>/dev/null" 2>/dev/null | tr -dc 0-9 || true)
  session=$($A800A "tmux has-session -t '$RUN_ID' 2>/dev/null && echo ALIVE || echo GONE" 2>/dev/null || echo UNKNOWN)
  cur_step=$($A800A "tail -n 6 '$TRAIN_LOG' 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/${STEP}' | grep -oE '^[0-9]+' | tail -1" 2>/dev/null | tr -dc 0-9 || true)
  [ -n "${cur_step:-}" ] && [ "${cur_step:-0}" -gt "$last_seen_step" ] && { last_seen_step=$cur_step; relaunch_count=0; }

  # COMPLETE: final ckpt exists (flow_only -> NO eval; stop + await stage-2 decision)
  if [ -n "${remote_sz:-}" ] && [ "${remote_sz:-0}" -gt 1000000 ]; then
    mark "STAGE1_DONE step=$STEP (Utonia learn_scene_flow flow_only complete on a800a GPU1). Ckpt at $CKPT_FINAL. Awaiting manual stage-2 decision (no auto-progression)."
    exit 0
  fi
  # real code crash -> flag, do NOT auto-resume
  if [ -n "${crash:-}" ] && [ "${crash:-0}" != "0" ] && [ "$train_exit" != "TRAIN_EXIT=0" ]; then
    mark "CRASHED (code error) — $($A800A "grep -E 'Traceback|Error|out of memory' '$TRAIN_LOG' 2>/dev/null | tail -3" 2>/dev/null)"
    exit 0
  fi
  # SESSION GONE (pod restart / SIGKILL) -> auto-resume
  if [ "$session" = "GONE" ] && [ "$train_exit" != "TRAIN_EXIT=0" ]; then
    relaunch_count=$((relaunch_count + 1))
    if [ "$relaunch_count" -gt "$MAX_RELAUNCH" ]; then
      mark "GAVE_UP: $MAX_RELAUNCH relaunches without lasting progress (last_step=$last_seen_step) — manual check needed."
      exit 0
    fi
    any_ckpt=$($A800A "ls '$CKPT_DIR'/steps_*_pytorch_model.pt 2>/dev/null | head -1" 2>/dev/null || true)
    R=$([ -n "$any_ckpt" ] && echo 1 || echo 0)
    $A800A "cd '$A800_REPO' && bash scripts/a800/_resume_utonia_learnflow.sh $R" >> "$LOG" 2>&1 || true
    mark "$([ "$R" = 1 ] && echo RESUMED || echo RELAUNCHED_FRESH) (#$relaunch_count, last_step=$last_seen_step)"
    sleep 120; continue
  fi

  mark "RUNNING ${cur_step:-<pre-train>}/${STEP} (relaunches=$relaunch_count)"
  sleep "$POLL"
done
