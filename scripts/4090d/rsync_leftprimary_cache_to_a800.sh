#!/usr/bin/env bash
# rsync the freshly-built leftprimary FFS + Utonia caches from 4090d to a800 (shared NAS),
# landing them under the FIXED dir names that cached-leftprimary trainings read.
# a800a host; a800b shares the same /home NAS -> one rsync covers both machines.
# Integrity gate: every suite's meta.json must say leftprimary before we push (never ship a
# rightprimary/partial cache).
set -uo pipefail
cd /data/wangqiwei/ICLR2026/starVLA
A800_SSH="ssh -i /data/wangqiwei/.ssh/id_a800_push -p 30947 -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=15"
A800_HOST=wangqiwei@10.13.32.4
A800_DST=/home/wangqiwei/ICLR2026/starVLA/playground/Datasets

# source dir on 4090d -> fixed dir name on a800
declare -A SRC_DST=(
  [ffs_net0_cache_leftprimary_4090d]=ffs_net0_cache
  [utonia_cache_perpatch_leftprimary_4090d]=utonia_cache_perpatch
)

# ---- integrity gate: all 8 suite metas must be leftprimary ----
bad=0
for src in "${!SRC_DST[@]}"; do
  for s in object goal spatial 10; do
    m="playground/Datasets/${src}/libero_${s}_no_noops_1.0.0_lerobot/meta.json"
    if grep -q leftprimary "$m" 2>/dev/null; then :; else echo "[gate] BAD/MISSING ${src}/${s} meta ($m)"; bad=1; fi
  done
done
[ "$bad" = 0 ] || { echo "[gate] cache integrity FAILED — aborting rsync"; exit 3; }
echo "[gate] all 8 suite metas = leftprimary OK"

# ---- rsync (resumable; -a preserves; --info=progress2 for a single % line) ----
for src in "${!SRC_DST[@]}"; do
  dst="${SRC_DST[$src]}"
  echo "[rsync] ${src} ($(du -sh playground/Datasets/${src} 2>/dev/null|cut -f1)) -> a800:${A800_DST}/${dst}  $(date -u +%FT%TZ)"
  rsync -a --info=progress2 --partial -e "$A800_SSH" \
    "playground/Datasets/${src}/" "${A800_HOST}:${A800_DST}/${dst}/"
  echo "[rsync] ${dst} done $(date -u +%FT%TZ)"
done

# ---- verify a800 sizes + a spot-check meta on the remote ----
echo "[verify] a800 cache sizes + a meta spot-check:"
$A800_SSH "$A800_HOST" "du -sh ${A800_DST}/ffs_net0_cache ${A800_DST}/utonia_cache_perpatch 2>/dev/null; grep -o leftprimary ${A800_DST}/utonia_cache_perpatch/libero_object_no_noops_1.0.0_lerobot/meta.json 2>/dev/null | head -1"
echo "RSYNC_ALL_DONE $(date -u +%FT%TZ)"
