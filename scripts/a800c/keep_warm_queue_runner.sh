#!/usr/bin/env bash
# keep_warm_queue_runner.sh — the REMOTE watcher loop for /gpu-queue-keep-warm.
# Polls nvidia-smi; when a queued experiment's target GPU(s) are free, launches it in
# its own tmux session; reaps finished runs; optionally prunes each finished run's ckpt
# dir to the last K checkpoints. Keeps the box busy so the platform's idle-reclaim
# (>4h avg util <26%) never fires.
#
# ⚠️ UNTESTED until first real run — this needs a real GPU box (nvidia-smi + tmux) to
#    verify. Read it before trusting it; it is intentionally simple + heavily guarded.
#
# This script is meant to RUN ON the GPU host (deployed into the remote project tree,
# e.g. scripts/<host>/). It is started robustly from the Mac via launch-remote-watcher's
# launch_watcher.sh (tmux, anti-flaky-net). It manages GPUs on the SAME host it runs on.
# For multiple hosts, run ONE runner per host.
#
# If extended to manage other hosts, provide their endpoints through deployment-local
# environment variables; do not hardcode private addresses or workstation-only aliases.
#
# Config via env (all paths are on the REMOTE host):
#   QUEUE_FILE   (required)  queue file, lines:
#                "<launcher cmd> | <gpu or gpu,list> | <run_id> | <ckpt_dir> | <smoke_status>"
#                smoke_status must be PASS to auto-launch; anything else -> BLOCKED (not started).
#   REPO         (required)  remote repo root; runner cd's here, launcher paths are relative to it
#   STATE_DIR    state/logs dir (default $REPO/playground/Checkpoints/keepwarm_state)
#   POLL         poll seconds (default 120)
#   MEM_FREE_MIB a GPU counts as physically free if memory.used < this (default 1500)
#   LAUNCH_GRACE seconds a freshly-launched run/GPU is protected from reap+reuse (default 300)
#   PRUNE_MODE   off | dryrun | apply  (default off) — 'apply' deletes; user must pre-approve
#   PRUNE_GLOB   ckpt subdir glob under ckpt_dir (default 'checkpoint-*')
#   KEEP_CKPTS   how many newest ckpts to keep when pruning (default 2)
#   FINAL_CKPT_GLOB checkpoint evidence accepted as a final ckpt (default 'checkpoint-final* final*')
#
# Reaping: when a run's tmux session is gone (past grace), it is DONE+pruned ONLY if it
#   exited 0 AND produced a final checkpoint (or a DONE/final marker); otherwise it is
#   recorded FAILED and NOT pruned — so a crash/OOM/instant-exit can't be mistaken for
#   success and delete good ckpts. Each launch records its exit code to $STATE_DIR/<rid>.exit.
# Singleton: an atomic mkdir lock ($STATE_DIR/runner.lock) ensures one runner per state dir.
#
# ⚠️ Prune relies on GNU `head -n -K` (keep-all-but-last-K). Only valid on remote Linux;
#    non-GNU head (mac) lacks negative -n and silently degrades to NOT deleting
#    (fail-closed = safe). Run prune only on the GPU Linux box, never locally.
#
# Clean stop:  touch $STATE_DIR/STOP   (runner exits at next loop top; no pkill needed)
set -uo pipefail

QUEUE_FILE="${QUEUE_FILE:?need QUEUE_FILE}"
REPO="${REPO:?need REPO}"
STATE_DIR="${STATE_DIR:-$REPO/playground/Checkpoints/keepwarm_state}"
POLL="${POLL:-120}"
MEM_FREE_MIB="${MEM_FREE_MIB:-1500}"
LAUNCH_GRACE="${LAUNCH_GRACE:-300}"
PRUNE_MODE="${PRUNE_MODE:-off}"
PRUNE_GLOB="${PRUNE_GLOB:-checkpoint-*}"
KEEP_CKPTS="${KEEP_CKPTS:-2}"
FINAL_CKPT_GLOB="${FINAL_CKPT_GLOB:-checkpoint-final* final*}"

mkdir -p "$STATE_DIR"

# ---- singleton lock: one runner per state dir (atomic mkdir) ----
LOCK="$STATE_DIR/runner.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[keepwarm] another runner holds $LOCK (pid $(cat "$LOCK/pid" 2>/dev/null)) — exiting" >&2
  exit 1
fi
echo "$$" > "$LOCK/pid"
trap 'rm -f "$LOCK/pid" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT

RUNNING="$STATE_DIR/running.tsv"   # run_id<TAB>gpu<TAB>started_epoch<TAB>ckpt_dir
LAUNCHED="$STATE_DIR/launched.list" # run_ids ever launched (one per line)
DONE="$STATE_DIR/done.tsv"          # run_id<TAB>finished_epoch<TAB>DONE
FAILED="$STATE_DIR/failed.tsv"      # run_id<TAB>finished_epoch<TAB>reason  (NOT pruned)
BLOCKED_LIST="$STATE_DIR/blocked.list" # run_ids blocked (smoke_status != PASS); dedup log
STATUS="$STATE_DIR/status.txt"      # human-readable snapshot, rewritten each loop
touch "$RUNNING" "$LAUNCHED" "$DONE" "$FAILED" "$BLOCKED_LIST"

log(){ echo "[keepwarm $(date '+%F %T')] $*"; }
trim(){ printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

# mem.used for one GPU index (MiB, integer), or "" if unknown
gpu_mem(){
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v g="$1" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if($1==g) print $2}'
}
# avg util across all GPUs (for the status snapshot / anti-reclaim sanity)
avg_util(){
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk '{s+=$1; n++} END{if(n)printf "%d", s/n; else print "?"}'
}
# is a comma-list of GPUs physically free (every member below mem threshold)?
gpus_free(){
  local list="$1" g m
  for g in $(echo "$list" | tr ',' ' '); do
    m="$(gpu_mem "$g")"
    [ -z "$m" ] && return 1            # unknown -> treat as not-free (fail-closed)
    [ "$m" -ge "$MEM_FREE_MIB" ] && return 1
  done
  return 0
}
# is any GPU in the list already claimed by one of OUR running entries?
gpus_claimed(){
  local list="$1" g claimed
  # column 2 of every running line, comma+newline -> spaces, padded so we can match " g "
  claimed=" $(cut -f2 "$RUNNING" 2>/dev/null | tr ',\n' '  ') "
  for g in $(echo "$list" | tr ',' ' '); do
    case "$claimed" in *" $g "*) return 0;; esac
  done
  return 1
}

prune_run(){  # $1=run_id $2=ckpt_dir
  local rid="$1" cd="$2" keep="$KEEP_CKPTS"
  [ -z "$cd" ] && return 0
  [ "$PRUNE_MODE" = "off" ] && return 0
  [ -d "$REPO/$cd" ] || { [ -d "$cd" ] || { log "prune skip $rid: ckpt_dir not found ($cd)"; return 0; }; }
  local base="$cd"; [ -d "$REPO/$cd" ] && base="$REPO/$cd"
  # newest-last by trailing integer (checkpoint-1000 < checkpoint-20000)
  local all victims
  all="$(ls -d "$base"/$PRUNE_GLOB 2>/dev/null | sed 's#/$##' | awk -F'[^0-9]+' '{print $(NF)" "$0}' | sort -n | awk '{print $2}')"
  [ -z "$all" ] && { log "prune skip $rid: no ckpts match $PRUNE_GLOB in $base"; return 0; }
  victims="$(printf '%s\n' "$all" | head -n -"$keep" 2>/dev/null)"
  [ -z "$victims" ] && { log "prune $rid: <= $keep ckpts, nothing to drop"; return 0; }
  if [ "$PRUNE_MODE" = "apply" ]; then
    printf '%s\n' "$victims" | while IFS= read -r d; do [ -n "$d" ] && rm -rf -- "$d" && log "pruned $d"; done
  else
    printf '%s\n' "$victims" | while IFS= read -r d; do [ -n "$d" ] && log "DRYRUN would prune $d"; done
  fi
}

# did a finished run actually SUCCEED? success = captured exit 0 AND evidence of a
# final checkpoint (or explicit max_steps marker). Anything else (non-zero exit,
# missing exit file = killed/OOM/crash, exit 0 but no ckpt = instant no-op) -> FAILED.
# Logs the specific failure reason; prints nothing on success. Returns 0 = succeeded.
run_succeeded(){  # $1=run_id  $2=ckpt_dir
  local rid="$1" cd="$2" ec base
  ec="$(cat "$STATE_DIR/$rid.exit" 2>/dev/null)"
  case "$ec" in
    0) : ;;
    '') log "run '$rid': no exit code captured (killed/OOM/crash before record) -> FAILED"; return 1;;
    *)  log "run '$rid': exited non-zero (code=$ec) -> FAILED"; return 1;;
  esac
  # exit 0 — require a real final marker/checkpoint so an instant or early exit can't fake success.
  # Any ordinary checkpoint-* is NOT enough: early-crashed runs often leave one.
  if [ -n "$cd" ]; then
    base="$cd"; [ -d "$REPO/$cd" ] && base="$REPO/$cd"
    if [ -d "$base" ]; then
      if [ -e "$base/DONE" ] || [ -e "$base/final" ]; then return 0; fi
      local pat
      for pat in $FINAL_CKPT_GLOB; do
        if ls -d "$base"/$pat >/dev/null 2>&1; then return 0; fi
      done
      log "run '$rid': exit 0 but no final marker/checkpoint in $base -> FAILED (suspect early exit; ordinary $PRUNE_GLOB is insufficient)"
      return 1
    fi
    log "run '$rid': exit 0 but ckpt_dir missing ($cd) -> FAILED"
    return 1
  fi
  return 0   # no ckpt_dir to verify against -> trust exit 0
}

log "runner up. repo=$REPO queue=$QUEUE_FILE poll=${POLL}s prune=$PRUNE_MODE keep=$KEEP_CKPTS"
cd "$REPO" 2>/dev/null || { log "FATAL cannot cd $REPO"; exit 1; }
[ -f "$QUEUE_FILE" ] || { log "FATAL queue file not found: $QUEUE_FILE"; exit 1; }
if ! grep -qE '^[[:space:]]*[^#[:space:]]' "$QUEUE_FILE"; then
  log "FATAL queue file has no usable (non-comment) lines: $QUEUE_FILE"; exit 1
fi

while :; do
  [ -f "$STATE_DIR/STOP" ] && { log "STOP file present — exiting cleanly"; exit 0; }
  now="$(date +%s)"

  # ---- reap finished runs (tmux session gone after grace) ----
  tmp="$RUNNING.tmp"; : > "$tmp"
  while IFS=$'\t' read -r rid gpu started ckpt; do
    [ -z "${rid:-}" ] && continue
    age=$(( now - started ))
    if tmux has-session -t "kw_$rid" 2>/dev/null; then
      printf '%s\t%s\t%s\t%s\n' "$rid" "$gpu" "$started" "$ckpt" >> "$tmp"   # still alive
    elif [ "$age" -lt "$LAUNCH_GRACE" ]; then
      printf '%s\t%s\t%s\t%s\n' "$rid" "$gpu" "$started" "$ckpt" >> "$tmp"   # too young to reap
    elif run_succeeded "$rid" "$ckpt"; then
      log "run '$rid' (gpu $gpu) DONE after ${age}s (exit 0 + final ckpt)"
      printf '%s\t%s\tDONE\n' "$rid" "$now" >> "$DONE"
      prune_run "$rid" "$ckpt"
    else
      log "run '$rid' (gpu $gpu) FAILED after ${age}s — NOT pruning (ckpts kept for inspection)"
      printf '%s\t%s\tFAILED\n' "$rid" "$now" >> "$FAILED"
    fi
  done < "$RUNNING"
  mv "$tmp" "$RUNNING"

  # ---- launch next eligible queued entries onto free GPUs ----
  while IFS='|' read -r f_launch f_gpu f_run f_ckpt f_smoke; do
    case "$(trim "${f_launch:-}")" in ''|'#'*) continue;; esac
    launch="$(trim "$f_launch")"; gpu="$(trim "${f_gpu:-}")"
    rid="$(trim "${f_run:-}")";   ckpt="$(trim "${f_ckpt:-}")"
    smoke="$(trim "${f_smoke:-}")"
    # run_id is a tmux session name + state file key -> fail-closed on bad charset
    case "$rid" in
      ''|*[!A-Za-z0-9._-]*) log "bad queue line (run_id must be [A-Za-z0-9._-], got '$rid'): $f_launch"; continue;;
    esac
    [ -z "$gpu" ] && { log "bad queue line (need gpu): $f_launch"; continue; }
    grep -qxF "$rid" "$LAUNCHED" && continue            # already launched once -> never relaunch
    # smoke gate: only launchers marked PASS auto-start; others stay BLOCKED (logged once)
    case "$smoke" in
      PASS|pass|Pass) : ;;
      *)
        if ! grep -qxF "$rid" "$BLOCKED_LIST" 2>/dev/null; then
          echo "$rid" >> "$BLOCKED_LIST"
          log "BLOCKED '$rid': smoke_status='$smoke' (need PASS) — not auto-launching"
        fi
        continue;;
    esac
    gpus_claimed "$gpu" && continue                     # one of our runs holds this GPU
    gpus_free "$gpu" || continue                        # someone (us earlier / external) busy
    sess="kw_$rid"
    rm -f "$STATE_DIR/$rid.exit"                         # clear stale exit code before (re)launch
    tmux new-session -d -s "$sess" \
      "cd '$REPO' && CUDA_VISIBLE_DEVICES=$gpu bash $launch >> '$STATE_DIR/$rid.log' 2>&1; echo \$? > '$STATE_DIR/$rid.exit'" 2>/dev/null
    sleep 3
    if tmux has-session -t "$sess" 2>/dev/null; then
      echo "$rid" >> "$LAUNCHED"
      printf '%s\t%s\t%s\t%s\n' "$rid" "$gpu" "$now" "$ckpt" >> "$RUNNING"
      log "LAUNCHED '$rid' on gpu $gpu (session $sess): $launch"
    else
      log "launch of '$rid' did not stick (tmux session died) — will retry next loop"
    fi
  done < "$QUEUE_FILE"

  # ---- status snapshot (Codex mirrors this into active-session memory) ----
  {
    echo "=== keepwarm status @ $(date '+%F %T')  avg_util=$(avg_util)% ==="
    echo "repo=$REPO  poll=${POLL}s  prune=$PRUNE_MODE keep=$KEEP_CKPTS"
    echo "-- running --"; cat "$RUNNING" 2>/dev/null
    echo "-- done --";    cat "$DONE" 2>/dev/null
    echo "-- failed (kept, NOT pruned) --"; cat "$FAILED" 2>/dev/null
    echo "-- blocked (smoke != PASS) --";   cat "$BLOCKED_LIST" 2>/dev/null
    echo "-- queue (run_ids) --"; awk -F'|' '!/^[[:space:]]*#/ && NF>=3 {gsub(/ /,"",$3); if($3!="")print $3}' "$QUEUE_FILE"
  } > "$STATUS"

  sleep "$POLL"
done
