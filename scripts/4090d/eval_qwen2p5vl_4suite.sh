#!/usr/bin/env bash
# Eval a ckpt on all 4 LIBERO suites. num_obs_frames / video_keys / observation_indices ALL derived
# from the ckpt's DataConfig (single source of truth): config.full.yaml data_mix -> mixture ->
# robot_type -> ROBOT_TYPE_CONFIG_MAP[rt].{observation_indices, video_keys}. Supports any frame
# stride (e.g. [-4,-2,0]). Caller-passed VIDEO_KEYS ($3) must match derived else FATAL.
# Usage: eval_qwen2p5vl_4suite.sh <RUN_ID> <STEP> [VIDEO_KEYS] [GPUS]
set -uo pipefail

RUN_ID="${1:?need RUN_ID}"; STEP="${2:?need STEP}"
VIDEO_KEYS_ARG="${3:-}"
GPUS="${4:-1 3 4 6}"

STARVLA=/data/wangqiwei/ICLR2026/starVLA
H100B_SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=90 -o ServerAliveInterval=15 root@10.112.2.93"
H100B_RUN=/mnt/data/wangqiwei/wangqiwei/starVLA/playground/Checkpoints/${RUN_ID}
LOCAL_RUN=${STARVLA}/playground/Checkpoints/${RUN_ID}
CKPT=${LOCAL_RUN}/checkpoints/steps_${STEP}_pytorch_model.pt

# ---- 1. transfer ckpt + config + stats from h100b ----
mkdir -p "${LOCAL_RUN}/checkpoints"
REMOTE_SZ=$($H100B_SSH "stat -c %s ${H100B_RUN}/checkpoints/steps_${STEP}_pytorch_model.pt" 2>/dev/null | tr -dc 0-9)
LOCAL_SZ=$( [ -f "$CKPT" ] && stat -c %s "$CKPT" || echo 0 )
if [ "${REMOTE_SZ:-0}" = "0" ]; then
  if [ "${LOCAL_SZ:-0}" -gt 1000000 ] && [ -f "${LOCAL_RUN}/config.yaml" ] && [ -f "${LOCAL_RUN}/dataset_statistics.json" ]; then
    echo "[xfer] h100b unreachable; using existing LOCAL ckpt+config+stats ($(du -h "$CKPT"|cut -f1))"; REMOTE_SZ="$LOCAL_SZ"
  else
    echo "[FATAL] remote ckpt steps_${STEP} not on h100b AND no complete local copy"; exit 1
  fi
fi
if [ "$LOCAL_SZ" != "$REMOTE_SZ" ]; then
  echo "[xfer] pulling steps_${STEP} ($((REMOTE_SZ/1000000))MB) from h100b ..."
  $H100B_SSH "cat ${H100B_RUN}/checkpoints/steps_${STEP}_pytorch_model.pt" > "$CKPT"
  $H100B_SSH "cat ${H100B_RUN}/config.full.yaml" > "${LOCAL_RUN}/config.full.yaml"
  $H100B_SSH "cat ${H100B_RUN}/dataset_statistics.json" > "${LOCAL_RUN}/dataset_statistics.json"
  cp "${LOCAL_RUN}/config.full.yaml" "${LOCAL_RUN}/config.yaml"
  NEW_SZ=$(stat -c %s "$CKPT")
  [ "$NEW_SZ" = "$REMOTE_SZ" ] || { echo "[FATAL] ckpt size mismatch local=$NEW_SZ remote=$REMOTE_SZ"; exit 2; }
  echo "[xfer] ckpt OK $(du -h "$CKPT"|cut -f1); config+stats pulled"
else
  echo "[xfer] ckpt already local + size matches, skip"
fi

# ---- 1b. derive observation_indices + video_keys (single source = DataConfig; fail-closed) ----
CFG="${LOCAL_RUN}/config.full.yaml"
_DERIVED=$(${STARVLA}/.venv/bin/python - "$CFG" "$STARVLA" <<'PYEOF'
import sys, yaml
cfg_path, starvla = sys.argv[1], sys.argv[2]
sys.path.insert(0, starvla)
try:
    cfg = yaml.safe_load(open(cfg_path)); dm = cfg["datasets"]["vla_data"]["data_mix"]
except Exception as e:
    sys.stderr.write("config/data_mix unreadable: %s\n" % e); sys.exit(11)
try:
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
    entries = DATASET_NAMED_MIXTURES[dm]
    dcs = [ROBOT_TYPE_CONFIG_MAP[e[2]] for e in entries]
    obs_set = set(tuple(int(i) for i in d.observation_indices) for d in dcs)
    vk_set = set(tuple(d.video_keys) for d in dcs)
    if len(obs_set) != 1 or len(vk_set) != 1:
        sys.stderr.write("mixture %s inconsistent across entries: obs=%s vk=%s\n" % (dm, obs_set, vk_set)); sys.exit(13)
    obs = list(next(iter(obs_set)))
    vks = list(next(iter(vk_set)))
except Exception as e:
    sys.stderr.write("dataconfig resolve failed for data_mix=%s: %s\n" % (dm, e)); sys.exit(12)
cammap = {"primary_image": "primary", "wrist_image": "wrist", "right_view": "right_view"}
cams = [cammap.get(k.split(".")[-1], k.split(".")[-1]) for k in vks]
print("%d|%s|%s" % (len(obs), ",".join(cams), ",".join(str(i) for i in obs)))
PYEOF
) || { echo "[FATAL] obs_indices/video_keys derivation failed -- refusing to eval"; exit 7; }
NUM_OBS_FRAMES=$(echo "$_DERIVED" | cut -d'|' -f1)
VK_DERIVED=$(echo "$_DERIVED" | cut -d'|' -f2)
OBS_INDICES=$(echo "$_DERIVED" | cut -d'|' -f3)
if [ -n "$VIDEO_KEYS_ARG" ] && [ "$VIDEO_KEYS_ARG" != "$VK_DERIVED" ]; then
  echo "[FATAL] passed VIDEO_KEYS='$VIDEO_KEYS_ARG' != derived '$VK_DERIVED' -- refusing"; exit 8
fi
VIDEO_KEYS="$VK_DERIVED"
echo "[mf] obs_indices=$OBS_INDICES video_keys=$VIDEO_KEYS num_frames=$NUM_OBS_FRAMES (from DataConfig, single source)"

# ---- 2. eval over 4 suites, 1 server / GPU, batched ----
SUITES=(libero_spatial libero_object libero_goal libero_10)
GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}; BASE_PORT=6730
[ "$NG" -ge 1 ] || { echo "[FATAL] no GPUs given"; exit 3; }
EVAL_ROOT=${LOCAL_RUN}/eval_step${STEP}_$(echo "$VIDEO_KEYS" | tr ',' '_')

i=0; LANE_FAIL=0
while [ $i -lt ${#SUITES[@]} ]; do
  pids=(); names=()
  for ((j=0; j<NG && i<${#SUITES[@]}; j++, i++)); do
    S=${SUITES[$i]}; G=${GPU_ARR[$j]}; P=$((BASE_PORT+i))
    ED=${EVAL_ROOT}/${S}
    rm -rf "$ED"; mkdir -p "$ED"
    echo "[eval] ${S} on GPU${G} port${P} -> ${ED}"
    bash "${STARVLA}/scripts/4090d/eval_one_ckpt.sh" "$CKPT" "$G" "$P" "$ED" "$S" "$VIDEO_KEYS" "$NUM_OBS_FRAMES" "$OBS_INDICES" > "${ED}/lane.log" 2>&1 &
    pids+=($!); names+=("$S")
    sleep 8
  done
  echo "[eval] batch launched (${#pids[@]} lanes, 1/GPU), waiting ..."
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then LANE_FAIL=$((LANE_FAIL+1)); echo "[eval] lane ${names[$idx]} exited non-zero"; fi
  done
done

# ---- 3. collect SR ----
echo "===== RESULTS RUN_ID=${RUN_ID} step=${STEP} video_keys=${VIDEO_KEYS} obs_indices=${OBS_INDICES} ====="
OK=0
for S in "${SUITES[@]}"; do
  ED=${EVAL_ROOT}/${S}
  SR=$(grep -iE "Total success rate" "${ED}/client.log" 2>/dev/null | grep -oE "[0-9]+\.?[0-9]*" | tail -1)
  echo "  ${S}: SR=${SR:-FAILED}  (log ${ED}/client.log)"
  [ -n "$SR" ] && OK=$((OK+1))
done
echo "ALL_EVAL_DONE ${RUN_ID} step${STEP} ${VIDEO_KEYS}"

STRAGGLERS=$(ps -eo pid,cmd | grep -E "server_policy\.py|eval_libero\.py" | grep -F "$CKPT" | grep -v grep | awk '{print $1}')
[ -n "$STRAGGLERS" ] && { echo "[cleanup] kill stragglers step${STEP}: $STRAGGLERS"; kill -9 $STRAGGLERS 2>/dev/null || true; }

if [ "$LANE_FAIL" -gt 0 ] || [ "$OK" -lt 4 ]; then
  echo "[FATAL] eval incomplete: lane_fail=$LANE_FAIL ok_suites=$OK/4"; exit 9
fi
exit 0
