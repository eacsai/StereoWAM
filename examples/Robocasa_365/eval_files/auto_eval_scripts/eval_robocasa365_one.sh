#!/usr/bin/env bash
# Run one RoboCasa-365 task against an already-running starVLA policy server.
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  bash $0 <ckpt> <env_name> <horizon> <gpu_id> <port>

Environment overrides:
  ROBOCASA365_PYTHON  Python executable for the robocasa environment
  N_EPISODES          Number of rollouts, default 50
  N_ACT               Number of action steps, default 8
  LOG_DIR             Per-task log directory, default <ckpt>.eval/logs
  VID_DIR             Video output directory, default <ckpt>.eval/videos/<env>
USAGE
}

if [[ $# -ne 5 ]]; then
  usage >&2
  exit 2
fi

CKPT=$1
ENV_NAME=$2
HORIZON=$3
GPU_ID=$4
PORT=$5

N_EPISODES=${N_EPISODES:-50}
N_ACT=${N_ACT:-8}
ROBOCASA365_PYTHON=${ROBOCASA365_PYTHON:-/data/wangqiwei/ICLR2026/robocasa/.venv/bin/python}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

EVAL_DIR=${EVAL_DIR:-"$(python3 -c 'import sys,pathlib; print(pathlib.Path(sys.argv[1]).with_suffix(".eval"))' "${CKPT}")"}

SAFE_ENV=${ENV_NAME//\//_}
TASK_LOG_DIR=${LOG_DIR:-"${EVAL_DIR}/logs"}
TASK_VID_DIR=${VID_DIR:-"${EVAL_DIR}/videos/${SAFE_ENV}"}
mkdir -p "${TASK_LOG_DIR}" "${TASK_VID_DIR}"

LOG_FILE="${TASK_LOG_DIR}/client_${SAFE_ENV}_gpu${GPU_ID}_port${PORT}.log"

echo "[one] ckpt=${CKPT}"
echo "[one] env=${ENV_NAME} horizon=${HORIZON} gpu=${GPU_ID} port=${PORT} episodes=${N_EPISODES} n_act=${N_ACT}"
echo "[one] log=${LOG_FILE}"
echo "[one] video_dir=${TASK_VID_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
MUJOCO_GL="${MUJOCO_GL:-egl}" \
MUJOCO_EGL_DEVICE_ID="${GPU_ID}" \
"${ROBOCASA365_PYTHON}" -m examples.Robocasa_365.eval_files.simulation_env \
  --args.pretrained-path "${CKPT}" \
  --args.env-name "${ENV_NAME}" \
  --args.host 127.0.0.1 \
  --args.port "${PORT}" \
  --args.n-episodes "${N_EPISODES}" \
  --args.n-envs 1 \
  --args.max-episode-steps "${HORIZON}" \
  --args.n-action-steps "${N_ACT}" \
  --args.video-out-path "${TASK_VID_DIR}"
