#!/usr/bin/env bash
# FFS 4-suite eval on 4090d. Thin wrapper over eval_qwen2p5vl_4suite.sh that adds the ONE
# thing FFS runs need: override ffs_model_path in the ckpt config to the 4090d
# Fast-FoundationStereo path, since eval runs on 4090d.
#
# The ckpt/config/stat files must already be local in playground/Checkpoints/<RUN_ID>.
# Source-machine pulls are handled by dedicated watcher/sync scripts before this wrapper.
#
# Usage: eval_ffs_4suite.sh <RUN_ID> <STEP> [GPUS]
set -uo pipefail

RUN_ID="${1:?need RUN_ID}"; STEP="${2:?need STEP}"
GPUS="${3:-}"

STARVLA=/data/wangqiwei/ICLR2026/starVLA
CORE=${STARVLA}/scripts/4090d/eval_qwen2p5vl_4suite.sh
LOCAL_RUN=${STARVLA}/playground/Checkpoints/${RUN_ID}
CKPT=${LOCAL_RUN}/checkpoints/steps_${STEP}_pytorch_model.pt
FFS_4090D=/data/wangqiwei/ICLR2026/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth

[ -f "$FFS_4090D" ] || { echo "[FATAL] 4090d FFS weights missing: $FFS_4090D"; exit 3; }
export FFS_REPO_DIR=/data/wangqiwei/ICLR2026/Fast-FoundationStereo   # framework imports FoundationStereo from here

# ---- 1. require local ckpt + config + stats; no remote endpoint wait/pull here ----
[ -s "$CKPT" ] || { echo "[FATAL] local FFS ckpt missing or empty: $CKPT"; exit 1; }
if [ ! -s "${LOCAL_RUN}/config.full.yaml" ] && [ ! -s "${LOCAL_RUN}/config.yaml" ]; then
  echo "[FATAL] local config missing: ${LOCAL_RUN}/config.full.yaml or config.yaml"; exit 1
fi
[ -s "${LOCAL_RUN}/dataset_statistics.json" ] || { echo "[FATAL] local dataset_statistics.json missing: ${LOCAL_RUN}/dataset_statistics.json"; exit 1; }
if [ ! -s "${LOCAL_RUN}/config.full.yaml" ] && [ -s "${LOCAL_RUN}/config.yaml" ]; then
  cp "${LOCAL_RUN}/config.yaml" "${LOCAL_RUN}/config.full.yaml"
fi

# ---- 2. ⭐ override ffs_model_path to 4090d in BOTH config files the server may read ----
cp "${LOCAL_RUN}/config.full.yaml" "${LOCAL_RUN}/config.yaml"
sed -i "s|ffs_model_path:.*|ffs_model_path: ${FFS_4090D}|g" "${LOCAL_RUN}/config.full.yaml" "${LOCAL_RUN}/config.yaml"
echo "[ffs] ffs_model_path overridden -> ${FFS_4090D}"
# Verify the override actually landed in BOTH config files the server may read.
# A failed sed (no ffs_model_path key, or sed-special chars in the path) must NOT
# silently fall through to a non-existent source-machine path -> wrong/empty
# disparity at eval time, invalidating the result without any error.
for _cfg in "${LOCAL_RUN}/config.full.yaml" "${LOCAL_RUN}/config.yaml"; do
  grep -qF "ffs_model_path: ${FFS_4090D}" "${_cfg}" || { echo "[FATAL] ffs_model_path override failed/absent in ${_cfg} (expected '${FFS_4090D}'; not an FFS run, or sed failed)"; exit 4; }
done
echo "[ffs] verified ffs_model_path = ${FFS_4090D} in both config files"

# ---- 3. delegate to proven core eval (skips re-pull, keeps our fixed config, derives
#         video_keys + single-frame from DataConfig). Pass GPUS through if given. ----
echo "[ffs] -> core eval (video_keys auto-derived from ckpt data_mix; leftprimary -> primary,left_view)"
if [ -n "$GPUS" ]; then
  exec bash "$CORE" "$RUN_ID" "$STEP" "" "$GPUS"
else
  exec bash "$CORE" "$RUN_ID" "$STEP" ""
fi
