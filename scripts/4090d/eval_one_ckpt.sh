#!/usr/bin/env bash
# Generic single-ckpt LIBERO eval lane: starVLA policy server (1 GPU) + LIBERO client (CPU + EGL mujoco).
# Replaces the hardcoded per-run launchers (eval_mono_step / eval_y_stereo / eval_ckpt_sweep) with one
# parameterized entrypoint, so parallel multi-GPU sweeps just launch N copies on different GPU/port pairs.
#
# Usage: eval_one_ckpt.sh <CKPT_PATH> <GPU> <PORT> <EVAL_DIR> [TASK_SUITE] [VIDEO_KEYS] [NUM_OBS_FRAMES] [OBS_INDICES]
#   CKPT_PATH  abs path to steps_<N>_pytorch_model.pt
#   GPU        CUDA device index dedicated to this lane's server
#   PORT       policy server port (unique per lane)
#   EVAL_DIR   output dir for server.log / client.log / videos
#   TASK_SUITE libero suite name           (default: libero_goal)
#   VIDEO_KEYS comma-separated camera keys  (default: EMPTY -> AUTO-DERIVED from the ckpt's data_mix; see below)
#   NUM_OBS_FRAMES  consecutive frames per camera (default 1; 3-frame ckpts MUST pass 3)
#
# ⚠️⚠️⚠️ HARD-WON LESSONS baked in as preflight guards below (do not remove) ⚠️⚠️⚠️
#   These three eval pitfalls each silently produced WRONG results before being guarded here:
#   (1) STEREO CAMERA ORDER — the big one (2026-06-25 incident, cost hours). Right-primary models
#       (data_mix *_rightprimary) MUST eval with `right_view,primary` (right-first), matching training
#       (data_config video_keys=[right_view, primary], FFS enforces right_view_idx=0). The OLD default
#       `primary,right_view` REVERSED the cameras -> wrong stereo geometry / cam_rope camera-IDs / FFS L-R
#       disparity -> small persistent action errors that COMPOUND on long-horizon libero_10 (0.51 vs correct
#       0.74), while short goal still looked fine at ~0.90 (so it was easy to misdiagnose as "data/model bad").
#       Clean leftprimary models derive `primary,left_view`; legacy rightprimary models derive
#       `right_view,primary`. Never hardcode a stereo camera default.
#   (2) attn_implementation — Qwen3.5-0.8B has NATIVE MIXED attention preserved ONLY under flash_attention_2;
#       the curated config.yaml often omits attn_implementation -> eval defaults to sdpa -> all-softmax build ->
#       ckpt key mismatch / FFS softmax-layer detection fails / load crash. We ensure config.yaml carries it.
#   (3) FFS model path — configs save the TRAINING machine's path (/home A800, /mnt h100b); eval runs on 4090d
#       where the weights live under /data. We repoint it + export FFS_REPO_DIR.
set -euo pipefail

# Cap CPU threads per eval process (FIX 2026-06-05): server(torch)+client(mujoco/numpy)
# otherwise each spawns ~1 thread/core on this 384-core shared box -> CPU oversubscription
# that hammers the shared machine. 8 threads is plenty for single-env sim + GPU-bound inference.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8

CKPT="$1"; GPU="$2"; PORT="$3"; EVAL_DIR="$4"; TASK_SUITE="${5:-libero_goal}"; VIDEO_KEYS="${6:-}"; NUM_OBS_FRAMES="${7:-1}"; OBS_INDICES="${8:-}"
# back-compat: derive stride-1 obs_indices from NUM_OBS_FRAMES if 8th arg omitted
if [ -z "$OBS_INDICES" ]; then OBS_INDICES="0"; for ((k=1; k<=NUM_OBS_FRAMES-1; k++)); do OBS_INDICES="-$k,$OBS_INDICES"; done; fi
STARVLA_DIR=/data/wangqiwei/ICLR2026/starVLA
LIBERO_VENV=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv/bin/python
LIBERO_HOME=/data/wangqiwei/ICLR2026/SSF/libero_sim_env/LIBERO
FFS_4090D=/data/wangqiwei/ICLR2026/Fast-FoundationStereo

[ -f "$CKPT" ] || { echo "[FATAL] ckpt not found: $CKPT"; exit 1; }
mkdir -p "$EVAL_DIR/videos"

# RUN_DIR = .../Checkpoints/<run_id>  (config.yaml / config.full.yaml live here)
RUN_DIR="$(dirname "$(dirname "$CKPT")")"

# ---- PREFLIGHT GUARD 2: ensure config.yaml carries attn_implementation (see lesson (2) above) ----
if [ -f "$RUN_DIR/config.yaml" ] && ! grep -q "attn_implementation" "$RUN_DIR/config.yaml" 2>/dev/null \
   && [ -f "$RUN_DIR/config.full.yaml" ] && grep -q "attn_implementation" "$RUN_DIR/config.full.yaml" 2>/dev/null; then
  cp "$RUN_DIR/config.yaml" "$RUN_DIR/config.yaml.preattn.bak" 2>/dev/null || true
  cp "$RUN_DIR/config.full.yaml" "$RUN_DIR/config.yaml"
  echo "  [preflight] config.yaml lacked attn_implementation -> copied config.full.yaml over it"
fi

# ---- PREFLIGHT GUARD 3: repoint FFS model path to this machine + export FFS_REPO_DIR (see lesson (3)) ----
export FFS_REPO_DIR="${FFS_REPO_DIR:-$FFS_4090D}"
for cfg in "$RUN_DIR/config.yaml" "$RUN_DIR/config.full.yaml"; do
  [ -f "$cfg" ] && sed -i \
    -e "s#/home/wangqiwei/ICLR2026/Fast-FoundationStereo#$FFS_4090D#g" \
    -e "s#/mnt/data/wangqiwei/wangqiwei/Fast-FoundationStereo#$FFS_4090D#g" "$cfg" 2>/dev/null || true
done

# ---- FFS stereo convention from data_mix, UNCONDITIONAL (codex 2026-06-26) ----
# Set even when VIDEO_KEYS is passed explicitly, so explicit-VIDEO_KEYS wrappers (eval_ffs_4suite.sh,
# eval_qwen2p5vl_4suite.sh) also export the right convention for old FFS ckpts (else clean-path refuse
# crash), and a leaked legacy_unrotate from a prior eval is reset for a clean leftprimary eval.
_DM_CONV="$(grep -m1 -E "^[[:space:]]*data_mix:" "$RUN_DIR/config.yaml" 2>/dev/null | sed -E 's/.*data_mix:[[:space:]]*//; s/[^a-zA-Z0-9_]//g')"
case "$_DM_CONV" in
  *leftprimary*)  export FFS_STEREO_CONVENTION=leftprimary ;;
esac

# ---- PREFLIGHT GUARD 1 ⭐: derive STEREO camera order from the ckpt's data_mix (see lesson (1) above) ----
# Only when the caller did NOT pass VIDEO_KEYS explicitly. Fail-closed on anything we can't map.
if [ -z "$VIDEO_KEYS" ]; then
  DM="$(grep -m1 -E "^[[:space:]]*data_mix:" "$RUN_DIR/config.yaml" 2>/dev/null | sed -E "s/.*data_mix:[[:space:]]*//; s/[\"' ]//g")"
  case "$DM" in
    *leftprimary*)  VIDEO_KEYS="primary,left_view"; export FFS_STEREO_CONVENTION=leftprimary ;;    # clean convention: primary first, logical left_view second
    *primaryright*) VIDEO_KEYS="primary,right_view" ;;   # left-first stereo (older runs)
    *primarywrist*) VIDEO_KEYS="primary,wrist" ;;        # primary + wrist
    *mono*)         VIDEO_KEYS="primary" ;;              # mono
    *) echo "  [FATAL] cannot derive video_keys from data_mix='$DM' (run_dir=$RUN_DIR)."
       echo "          Pass VIDEO_KEYS explicitly as arg 6 (e.g. right_view,primary | primary | primary,wrist)."; exit 6 ;;
  esac
  echo "  [preflight] auto-derived video_keys='$VIDEO_KEYS' from data_mix='$DM'"
fi

echo "=== eval $CKPT  GPU=$GPU port=$PORT suite=$TASK_SUITE video_keys=$VIDEO_KEYS  $(date) ==="

pkill -9 -f "server_policy.py.*--port $PORT" 2>/dev/null || true
sleep 3

cd "$STARVLA_DIR"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port "$PORT" --use_bf16 \
  > "$EVAL_DIR/server.log" 2>&1 &
SERVER_PID=$!

READY=0
SERVER_READY_ITERS="${SERVER_READY_ITERS:-90}"  # 90*2s=180s default; raise under GPU contention
for i in $(seq 1 "$SERVER_READY_ITERS"); do
  if grep -q "listening" "$EVAL_DIR/server.log" 2>/dev/null; then
    echo "  [server ready iter $i pid $SERVER_PID]"; READY=1; break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "  [FATAL] server died before ready, see $EVAL_DIR/server.log"; exit 2
  fi
  sleep 2
done
[ "$READY" = 1 ] || { echo "  [FATAL] server not ready in time"; kill -9 $SERVER_PID 2>/dev/null || true; exit 3; }

export LIBERO_HOME LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# Pin this lane's EGL render to its own GPU (FIX 2026-06-04): the server is CUDA_VISIBLE_DEVICES-pinned
# but the client's mujoco EGL render was not -> N parallel lanes all rendered on the default GPU0 (contention).
export MUJOCO_EGL_DEVICE_ID="$GPU"

set +e
$LIBERO_VENV examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port "$PORT" \
  --args.task-suite-name "$TASK_SUITE" --args.num-trials-per-task 10 --args.max-tasks -1 \
  --args.video-keys "$VIDEO_KEYS" \
  --args.num-obs-frames "$NUM_OBS_FRAMES" \
  --args.obs-indices "$OBS_INDICES" \
  --args.video-out-path "$EVAL_DIR/videos" \
  > "$EVAL_DIR/client.log" 2>&1
CLIENT_RC=$?
set -e
# fail-closed (codex round-2): do NOT swallow eval_libero.py multi-frame asserts/ValueErrors.
# Non-zero client AND no success line => hard fail (kill server, exit).
if [ "$CLIENT_RC" -ne 0 ] && ! grep -q "Total success rate" "$EVAL_DIR/client.log" 2>/dev/null; then
  echo "  [FATAL] eval client failed (rc=$CLIENT_RC, no success line) -- see $EVAL_DIR/client.log"
  grep -iE "ValueError|AssertionError|num.obs.frames|image_list len|duplicate" "$EVAL_DIR/client.log" 2>/dev/null | tail -5
  kill -9 $SERVER_PID 2>/dev/null || true
  exit 5
fi

echo "  [RESULT] $(grep 'Total success rate' "$EVAL_DIR/client.log" | tail -1)  ($(date))"
kill -9 $SERVER_PID 2>/dev/null || true
sleep 3
echo "=== lane done: $CKPT ==="
