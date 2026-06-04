#!/usr/bin/env bash
# render_stereo_suite.sh <suite> <gpu>
# Full stereo-dataset pipeline for ONE LIBERO suite, mirroring the libero_goal build:
#   1) regen_vanilla        -> vanilla HDF5 (re-sim states + 256 agentview/eye_in_hand, no-op filtered)
#   2) render_rightview     -> + obs/right_view_rgb (state-replay, parallel 6cm rig, 180-deg flip)
#   3) Any4LeRobot (patched stereo) -> LeRobot v2.1 (adds observation.images.right_view)
#   4) verify counts + gripper polarity {0,1}
# Idempotent-ish: step 2 skips existing per-task files; steps 1/3 overwrite.
set -euo pipefail

SUITE="${1:?usage: render_stereo_suite.sh <suite> <gpu>}"
GPU="${2:?need gpu id}"

ROOT=/data/wangqiwei/ICLR2026
RAW="$ROOT/data/${SUITE}"
VAN="$ROOT/data/${SUITE}_openvla_vanilla"
STE="$ROOT/data/${SUITE}_openvla_vanilla_stereo"
LEROBOT_PARENT="$ROOT/data"
LEROBOT_OUT="$ROOT/data/${SUITE}_openvla_vanilla_stereo_lerobot"

RENDER_PY="$ROOT/SSF/libero_sim_env/.venv/bin/python"
CONV_PY="$ROOT/openvla_debug/.venv/bin/python"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_HOME="$ROOT/SSF/libero_sim_env/LIBERO"
export LIBERO_CONFIG_PATH="$LIBERO_HOME/libero"
export CUDA_VISIBLE_DEVICES="$GPU"

ts(){ date -u +%H:%M:%S; }
echo "[$(ts)] ===== render_stereo_suite SUITE=$SUITE GPU=$GPU ====="
test -d "$RAW" || { echo "[$(ts)] FATAL: raw demos missing: $RAW (download first)"; exit 1; }
echo "[$(ts)] raw demos: $(ls "$RAW"/*_demo.hdf5 2>/dev/null | wc -l) task files"

echo "[$(ts)] --- STEP1 regen_vanilla -> $VAN ---"
"$RENDER_PY" "$ROOT/openvla_debug/regen_vanilla_libero_goal.py" \
    --libero_task_suite "$SUITE" \
    --libero_raw_data_dir "$RAW" \
    --libero_target_dir "$VAN"
echo "[$(ts)] STEP1 done: $(ls "$VAN"/*_demo.hdf5 2>/dev/null | wc -l) vanilla hdf5"

echo "[$(ts)] --- STEP2 render_rightview -> $STE ---"
"$RENDER_PY" "$ROOT/openvla_debug/render_rightview_state_replay.py" \
    --suite "$SUITE" \
    --vanilla-dir "$VAN" \
    --out-dir "$STE"
echo "[$(ts)] STEP2 done: $(ls "$STE"/*_demo.hdf5 2>/dev/null | wc -l) stereo hdf5"

echo "[$(ts)] --- STEP3 Any4LeRobot convert -> $LEROBOT_OUT ---"
(
  cd "$ROOT/openvla_debug/libero2lerobot_patched_stereo"
  export HDF5_USE_FILE_LOCKING=FALSE HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE SVT_LOG=1
  "$CONV_PY" libero_h5.py \
      --src-paths "$STE" \
      --output-path "$LEROBOT_PARENT" \
      --executor local --tasks-per-job 3 --workers 10
)
echo "[$(ts)] STEP3 done (v3.0) -> $LEROBOT_OUT"

# libero_h5.py emits LeRobot v3.0 (consolidated parquet, NO episodes.jsonl). The starVLA
# gr00t_lerobot dataloader needs v2.1 (per-episode parquet + meta/episodes.jsonl). Convert in place.
echo "[$(ts)] --- STEP3.5 LeRobot v3.0 -> v2.1 (per-episode parquet + episodes.jsonl) ---"
(
  cd "$ROOT/openvla_debug/v30_to_v21_patched"
  "$CONV_PY" convert_dataset_v30_to_v21.py \
      --repo-id "local/${SUITE}_openvla_vanilla_stereo_lerobot" \
      --root "$LEROBOT_OUT"
)
echo "[$(ts)] STEP3.5 done (v2.1)"

echo "[$(ts)] --- STEP4 verify (v2.1 layout + counts + gripper polarity) ---"
"$CONV_PY" - "$LEROBOT_OUT" <<'PYV'
import sys, glob, json, os, numpy as np, pandas as pd
out = sys.argv[1]
pq = sorted(glob.glob(f"{out}/data/**/*.parquet", recursive=True))
ep_jsonl = os.path.join(out, "meta", "episodes.jsonl")
info = json.load(open(os.path.join(out, "meta", "info.json")))
print(f"  parquet files: {len(pq)} | total_episodes(info): {info.get('total_episodes')} | total_frames: {info.get('total_frames')}")
print(f"  episodes.jsonl: {'present (%d lines)' % sum(1 for _ in open(ep_jsonl)) if os.path.exists(ep_jsonl) else 'MISSING -- still v3.0!'}")
vk = [k for k in info.get('features', {}) if 'image' in k or 'view' in k]
print(f"  video keys: {vk}")
if not pq:
    print("  WARNING: no parquet produced"); sys.exit(0)
df = pd.concat([pd.read_parquet(p) for p in pq[:6]], ignore_index=True)
act = np.stack(df['action'].values); grip = act[:, -1]
ok = grip.min() >= -0.01 and grip.max() <= 1.01
print(f"  gripper last-col min/max: {grip.min():.3f}/{grip.max():.3f} -> POLARITY {'OK [0,1]' if ok else 'WARNING not in [0,1]!'}")
v21_ok = os.path.exists(ep_jsonl) and len(pq) > 1
print(f"  V2.1 LAYOUT: {'OK' if v21_ok else 'NOT v2.1 -- dataloader will fail'}")
PYV
echo "[$(ts)] ===== DONE SUITE=$SUITE -> $LEROBOT_OUT ====="
