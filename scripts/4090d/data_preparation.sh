#!/usr/bin/env bash
# data_preparation.sh — fetch IPEC LIBERO 4-suite + LLaVA-OneVision-COCO
# via aria2c (URL list generated from LeRobot layout). Replaces upstream
# examples/LIBERO/data_preparation.sh which:
#   1. would downgrade huggingface_hub to 0.35.3 (kills transformers)
#   2. relies on hf-hub's snapshot_download which can't use HF_ENDPOINT
#      mirror reliably (cursor pagination Link header points to hf.co)
#
# ⚠️  FAIL-CLOSED design (post 2026-05-19 codex review):
#   - `set -e` everywhere (no `|| echo …` after aria2)
#   - validate_suite()  checks EXACT N file count + nonzero size + no
#     .aria2 leftovers before printing OK
#   - validate_llava()  checks zip+json sizes + images/ non-empty
#   - skip-if-complete uses the same validator (no silent partial cert)
#
# Usage:
#   DEST=/data/wangqiwei/ICLR2026/data/libero_official bash scripts/4090d/data_preparation.sh
set -euo pipefail

DEST="${DEST:?set DEST}"
CUR=/data/wangqiwei/ICLR2026/starVLA
MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
cd "$CUR"

mkdir -p "$DEST/libero" "$DEST/LLaVA-OneVision-COCO"

declare -A SUITE_EPISODES=(
  [spatial]=432
  [object]=454
  [goal]=428
  [10]=379
)

# Per-suite expected meta files (4, no stats.json — none of the 4 IPEC suites ship it)
META_FILES=(info.json episodes.jsonl episodes_stats.jsonl tasks.jsonl)
MIN_PARQUET_BYTES=1024     # smallest real episode_*.parquet is ~10KB; 1KB is safe lower bound for "non-empty"
MIN_MP4_BYTES=1024
MIN_META_BYTES=8           # info.json ~3KB; tasks.jsonl can be a few hundred bytes; 8 just rejects zero-byte
LLAVA_ZIP_MIN_BYTES=2500000000   # 2.5 GB lower bound (HF says 2.56 GB)
LLAVA_JSON_MIN_BYTES=60000000    # 60 MB lower bound (HF says 68 MB)

# --------------------------------------------------------------------------
# validate_suite <suite> <out_dir> — return 0 iff suite is fully + cleanly downloaded
# --------------------------------------------------------------------------
validate_suite() {
  local suite="$1" out="$2"
  local N="${SUITE_EPISODES[$suite]}"
  local missing=0 small=0 leftover=0

  # check meta files
  local m
  for m in "${META_FILES[@]}"; do
    if [ ! -f "$out/meta/$m" ]; then
      echo "  [validate] MISSING meta/$m" >&2
      missing=$((missing+1))
    elif [ "$(stat -c %s "$out/meta/$m" 2>/dev/null || stat -f %z "$out/meta/$m")" -lt "$MIN_META_BYTES" ]; then
      echo "  [validate] TOO_SMALL meta/$m" >&2
      small=$((small+1))
    fi
  done

  # check per-episode parquet + 2 video files
  local i ep p v1 v2
  for i in $(seq 0 $((N-1))); do
    ep=$(printf "episode_%06d" "$i")
    p="$out/data/chunk-000/${ep}.parquet"
    v1="$out/videos/chunk-000/observation.images.image/${ep}.mp4"
    v2="$out/videos/chunk-000/observation.images.wrist_image/${ep}.mp4"
    for f in "$p" "$v1" "$v2"; do
      if [ ! -f "$f" ]; then
        missing=$((missing+1))
        [ "$missing" -le 3 ] && echo "  [validate] MISSING ${f#$out/}" >&2
      elif [ "$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")" -lt "$MIN_PARQUET_BYTES" ]; then
        small=$((small+1))
        [ "$small" -le 3 ] && echo "  [validate] TOO_SMALL ${f#$out/}" >&2
      fi
    done
  done

  # reject if any .aria2 control file remains (= file mid-download / corrupted)
  if find "$out" -name "*.aria2" -print -quit 2>/dev/null | grep -q .; then
    leftover=$(find "$out" -name "*.aria2" 2>/dev/null | wc -l | tr -d ' ')
    echo "  [validate] $leftover .aria2 control files left (download still in progress / failed mid-stream)" >&2
  fi

  if [ "$missing" -ne 0 ] || [ "$small" -ne 0 ] || [ "$leftover" -ne 0 ]; then
    echo "  [validate] FAIL suite=$suite : missing=$missing small=$small leftover=$leftover" >&2
    return 1
  fi
  return 0
}

# --------------------------------------------------------------------------
# validate_llava <dir> — return 0 iff LLaVA zip + json downloaded + unzipped non-empty
# --------------------------------------------------------------------------
validate_llava() {
  local llava="$1"
  local zip="$llava/sharegpt4v_coco.zip"
  local json="$llava/llava_jsons/sharegpt4v_coco.json"

  if [ ! -f "$zip" ] || [ "$(stat -c %s "$zip" 2>/dev/null || stat -f %z "$zip")" -lt "$LLAVA_ZIP_MIN_BYTES" ]; then
    echo "  [validate] LLaVA zip missing/too small" >&2; return 1
  fi
  if [ ! -f "$json" ] || [ "$(stat -c %s "$json" 2>/dev/null || stat -f %z "$json")" -lt "$LLAVA_JSON_MIN_BYTES" ]; then
    echo "  [validate] LLaVA json missing/too small" >&2; return 1
  fi
  if find "$llava" -name "*.aria2" -print -quit 2>/dev/null | grep -q .; then
    echo "  [validate] LLaVA .aria2 leftovers" >&2; return 1
  fi
  # post-unzip check (only if unzipped)
  if [ -d "$llava/images" ]; then
    local n=$(find "$llava/images" -type f 2>/dev/null | head -10 | wc -l | tr -d ' ')
    if [ "$n" -lt 1 ]; then
      echo "  [validate] LLaVA images/ dir empty" >&2; return 1
    fi
  fi
  return 0
}

# --------------------------------------------------------------------------
# main download loop
# --------------------------------------------------------------------------
for suite in spatial object goal 10; do
  N="${SUITE_EPISODES[$suite]}"
  repo="IPEC-COMMUNITY/libero_${suite}_no_noops_1.0.0_lerobot"
  out="$DEST/libero/libero_${suite}_no_noops_1.0.0_lerobot"

  # Skip if already validated complete
  if [ -d "$out/data/chunk-000" ] && validate_suite "$suite" "$out" 2>/dev/null; then
    echo "[suite] $suite already complete (validated); skip"
    continue
  fi

  echo "[suite] $suite ($N episodes) -> $out"
  mkdir -p "$out/data/chunk-000" \
           "$out/videos/chunk-000/observation.images.image" \
           "$out/videos/chunk-000/observation.images.wrist_image" \
           "$out/meta"

  # Generate URL list
  urls="/tmp/prep_libero_${suite}_urls.txt"
  : > "$urls"
  for f in "${META_FILES[@]}"; do
    {
      echo "${MIRROR}/datasets/${repo}/resolve/main/meta/${f}"
      echo "  out=meta/${f}"
    } >> "$urls"
  done
  for i in $(seq 0 $((N-1))); do
    ep=$(printf "episode_%06d" "$i")
    {
      echo "${MIRROR}/datasets/${repo}/resolve/main/data/chunk-000/${ep}.parquet"
      echo "  out=data/chunk-000/${ep}.parquet"
      echo "${MIRROR}/datasets/${repo}/resolve/main/videos/chunk-000/observation.images.image/${ep}.mp4"
      echo "  out=videos/chunk-000/observation.images.image/${ep}.mp4"
      echo "${MIRROR}/datasets/${repo}/resolve/main/videos/chunk-000/observation.images.wrist_image/${ep}.mp4"
      echo "  out=videos/chunk-000/observation.images.wrist_image/${ep}.mp4"
    } >> "$urls"
  done

  # Download — let aria2 propagate failure so set -e catches it.
  # aria2c exit codes: 0=ok, 3=resource not found, 7=unknown options, 24=auth failed.
  # We allow exit code 3 ONLY if validate_suite passes (sometimes a single 404
  # on a non-existent meta file is benign), so we capture exit code and
  # branch on validation instead of pipe-failing immediately.
  aria_rc=0
  aria2c \
    --dir="$out" \
    --input-file="$urls" \
    --max-concurrent-downloads=4 \
    --max-connection-per-server=2 \
    --split=2 \
    --min-split-size=1M \
    --continue=true \
    --auto-file-renaming=false \
    --conditional-get=true \
    --remote-time=true \
    --retry-wait=15 \
    --max-tries=20 \
    --timeout=120 \
    --connect-timeout=10 \
    --summary-interval=30 \
    --console-log-level=warn \
    --download-result=full || aria_rc=$?
  if [ "$aria_rc" -ne 0 ] && [ "$aria_rc" -ne 3 ]; then
    echo "[suite] $suite: aria2c exited with $aria_rc (NOT 0/3) — fail closed" >&2
    exit "$aria_rc"
  fi

  # Validate post-download (also catches the case "aria2c returned 3 but missed real files")
  if ! validate_suite "$suite" "$out"; then
    echo "[suite] $suite: validation FAILED after download — fail closed" >&2
    exit 1
  fi
  echo "[suite] OK $suite (validated: $N parquets + 2×$N mp4 + ${#META_FILES[@]} meta, no .aria2 leftovers)"
done

# --------------------------------------------------------------------------
# LLaVA-OneVision-COCO
# --------------------------------------------------------------------------
llava="$DEST/LLaVA-OneVision-COCO"
mkdir -p "$llava/llava_jsons"

if validate_llava "$llava" 2>/dev/null; then
  echo "[llava] already complete (validated); skip download"
else
  echo "[llava] downloading"
  urls="/tmp/prep_libero_llava_urls.txt"
  cat > "$urls" <<EOF
${MIRROR}/datasets/StarVLA/LLaVA-OneVision-COCO/resolve/main/sharegpt4v_coco.zip
  out=sharegpt4v_coco.zip
${MIRROR}/datasets/StarVLA/LLaVA-OneVision-COCO/resolve/main/llava_jsons/sharegpt4v_coco.json
  out=llava_jsons/sharegpt4v_coco.json
EOF
  aria_rc=0
  aria2c \
    --dir="$llava" \
    --input-file="$urls" \
    --max-concurrent-downloads=2 \
    --max-connection-per-server=2 \
    --split=2 \
    --min-split-size=1M \
    --continue=true \
    --auto-file-renaming=false \
    --remote-time=true \
    --retry-wait=15 \
    --max-tries=20 \
    --timeout=120 \
    --connect-timeout=10 \
    --console-log-level=warn \
    --download-result=full || aria_rc=$?
  if [ "$aria_rc" -ne 0 ]; then
    echo "[llava] aria2c exited with $aria_rc — fail closed" >&2
    exit "$aria_rc"
  fi
  if ! validate_llava "$llava"; then
    echo "[llava] validation FAILED — fail closed" >&2
    exit 1
  fi
  echo "[llava] OK (validated)"
fi

# Unzip if not already done
if [ ! -d "$llava/images" ] || [ -z "$(ls -A "$llava/images" 2>/dev/null)" ]; then
  echo "[unzip] sharegpt4v_coco.zip"
  unzip -o "$llava/sharegpt4v_coco.zip" -d "$llava/" > /tmp/prep_unzip.log
  echo "[unzip] done"
else
  echo "[unzip] already unzipped (images/ non-empty)"
fi
# re-validate post-unzip
if ! validate_llava "$llava"; then
  echo "[llava] post-unzip validation FAILED — fail closed" >&2
  exit 1
fi

# --------------------------------------------------------------------------
# Symlinks + modality.json (last step — only runs if everything above OK)
# --------------------------------------------------------------------------
mkdir -p "$CUR/playground/Datasets"
ln -sfn "$DEST/libero" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA"
ln -sfn "$DEST/LLaVA-OneVision-COCO" "$CUR/playground/Datasets/LLaVA-OneVision-COCO"

for suite in libero_10_no_noops_1.0.0_lerobot libero_goal_no_noops_1.0.0_lerobot libero_object_no_noops_1.0.0_lerobot libero_spatial_no_noops_1.0.0_lerobot; do
  cp "$CUR/examples/LIBERO/train_files/modality.json" \
     "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/${suite}/meta/modality.json"
done
echo "[modality.json] copied to all 4 suites"
echo "=== ALL DONE ==="
