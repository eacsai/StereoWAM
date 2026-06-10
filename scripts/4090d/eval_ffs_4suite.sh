#!/usr/bin/env bash
# FFS 4-suite eval on 4090d. Thin wrapper over eval_qwen2p5vl_4suite.sh that adds the ONE
# thing FFS runs need: override ffs_model_path (saved as the h100b /mnt/data path in the
# ckpt config) to the 4090d Fast-FoundationStereo path, since eval runs on 4090d.
#
# Everything else (ckpt pull from h100b, video_keys/obs_indices derivation from DataConfig,
# 4-suite server+client, SR collection) is reused verbatim from the proven core script.
#
# Usage: eval_ffs_4suite.sh <RUN_ID> <STEP> [GPUS]
set -uo pipefail

RUN_ID="${1:?need RUN_ID}"; STEP="${2:?need STEP}"
GPUS="${3:-}"

STARVLA=/data/wangqiwei/ICLR2026/starVLA
CORE=${STARVLA}/scripts/4090d/eval_qwen2p5vl_4suite.sh
H100B_SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=90 -o ServerAliveInterval=15 root@10.112.2.93"
H100B_RUN=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints/${RUN_ID}
LOCAL_RUN=${STARVLA}/playground/Checkpoints/${RUN_ID}
CKPT=${LOCAL_RUN}/checkpoints/steps_${STEP}_pytorch_model.pt
FFS_4090D=/data/wangqiwei/ICLR2026/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth

[ -f "$FFS_4090D" ] || { echo "[FATAL] 4090d FFS weights missing: $FFS_4090D"; exit 3; }
export FFS_REPO_DIR=/data/wangqiwei/ICLR2026/Fast-FoundationStereo   # framework imports FoundationStereo from here

# ---- 1. pull ckpt + config + stats from h100b (so core script sees ckpt size-match -> skips
#         its own pull -> keeps the config we are about to fix) ----
mkdir -p "${LOCAL_RUN}/checkpoints"
REMOTE_SZ=$($H100B_SSH "stat -c %s ${H100B_RUN}/checkpoints/steps_${STEP}_pytorch_model.pt" 2>/dev/null | tr -dc 0-9)
[ -n "${REMOTE_SZ:-}" ] && [ "${REMOTE_SZ:-0}" -gt 1000000 ] || { echo "[FATAL] FFS ckpt steps_${STEP} not on h100b ($H100B_RUN)"; exit 1; }
LOCAL_SZ=$( [ -f "$CKPT" ] && stat -c %s "$CKPT" || echo 0 )
if [ "$LOCAL_SZ" != "$REMOTE_SZ" ]; then
  echo "[ffs-xfer] pulling steps_${STEP} ($((REMOTE_SZ/1000000))MB) + config + stats from h100b ..."
  $H100B_SSH "cat ${H100B_RUN}/checkpoints/steps_${STEP}_pytorch_model.pt" > "$CKPT"
  $H100B_SSH "cat ${H100B_RUN}/config.full.yaml"          > "${LOCAL_RUN}/config.full.yaml"
  $H100B_SSH "cat ${H100B_RUN}/dataset_statistics.json"   > "${LOCAL_RUN}/dataset_statistics.json"
  NEW_SZ=$(stat -c %s "$CKPT")
  [ "$NEW_SZ" = "$REMOTE_SZ" ] || { echo "[FATAL] ckpt size mismatch local=$NEW_SZ remote=$REMOTE_SZ"; exit 2; }
  echo "[ffs-xfer] ckpt OK $(du -h "$CKPT"|cut -f1)"
else
  echo "[ffs-xfer] ckpt already local + size matches"
  # config may be stale/absent — re-pull config+stats to be safe (cheap).
  # Pull to .tmp and mv only on success: a direct `ssh > file || true` would
  # TRUNCATE the known-good local copy to empty on a transient h100b outage.
  if $H100B_SSH "cat ${H100B_RUN}/config.full.yaml" > "${LOCAL_RUN}/config.full.yaml.tmp" 2>/dev/null \
     && [ -s "${LOCAL_RUN}/config.full.yaml.tmp" ]; then
    mv "${LOCAL_RUN}/config.full.yaml.tmp" "${LOCAL_RUN}/config.full.yaml"
  else
    rm -f "${LOCAL_RUN}/config.full.yaml.tmp"
    echo "[ffs-xfer] WARN config re-pull failed; keeping existing local config.full.yaml"
  fi
  if $H100B_SSH "cat ${H100B_RUN}/dataset_statistics.json" > "${LOCAL_RUN}/dataset_statistics.json.tmp" 2>/dev/null \
     && [ -s "${LOCAL_RUN}/dataset_statistics.json.tmp" ]; then
    mv "${LOCAL_RUN}/dataset_statistics.json.tmp" "${LOCAL_RUN}/dataset_statistics.json"
  else
    rm -f "${LOCAL_RUN}/dataset_statistics.json.tmp"
    echo "[ffs-xfer] WARN stats re-pull failed; keeping existing local dataset_statistics.json"
  fi
fi

# ---- 2. ⭐ override ffs_model_path (h100b -> 4090d) in BOTH config files the server may read ----
cp "${LOCAL_RUN}/config.full.yaml" "${LOCAL_RUN}/config.yaml"
sed -i "s|ffs_model_path:.*|ffs_model_path: ${FFS_4090D}|g" "${LOCAL_RUN}/config.full.yaml" "${LOCAL_RUN}/config.yaml"
echo "[ffs] ffs_model_path overridden -> ${FFS_4090D}"
# Verify the override actually landed in BOTH config files the server may read.
# A failed sed (no ffs_model_path key, or sed-special chars in the path) must NOT
# silently fall through to the non-existent h100b path on this box -> wrong/empty
# disparity at eval time, invalidating the result without any error.
for _cfg in "${LOCAL_RUN}/config.full.yaml" "${LOCAL_RUN}/config.yaml"; do
  grep -qF "ffs_model_path: ${FFS_4090D}" "${_cfg}" || { echo "[FATAL] ffs_model_path override failed/absent in ${_cfg} (expected '${FFS_4090D}'; not an FFS run, or sed failed)"; exit 4; }
done
echo "[ffs] verified ffs_model_path = ${FFS_4090D} in both config files"

# ---- 3. delegate to proven core eval (skips re-pull, keeps our fixed config, derives
#         right_view,primary + single-frame from DataConfig). Pass GPUS through if given. ----
echo "[ffs] -> core eval (video_keys auto-derived; expect right_view,primary single-frame)"
if [ -n "$GPUS" ]; then
  exec bash "$CORE" "$RUN_ID" "$STEP" "right_view,primary" "$GPUS"
else
  exec bash "$CORE" "$RUN_ID" "$STEP" "right_view,primary"
fi
