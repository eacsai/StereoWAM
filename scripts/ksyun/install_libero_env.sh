#!/usr/bin/env bash
# Install LIBERO eval env in ksyun H100 dev container.
# Matches 4090d /data/wangqiwei/ICLR2026/SSF/libero_sim_env/.venv.
# Prereq: LIBERO repo already rsynced/streamed to /mnt/data/wangqiwei/wangqiwei/LIBERO/
set -euo pipefail

ENV_NAME="libero-eval"
LIBERO_REPO="/mnt/data/wangqiwei/wangqiwei/LIBERO"

source /opt/conda/etc/profile.d/conda.sh
if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "[1/3] creating conda env ${ENV_NAME} (py3.10)"
  conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"
python --version

# pip install (via ksyun pypi by default)
echo "[2/3] install LIBERO sim deps (mujoco/robosuite/gym/h5py/torch 2.12/tyro/websockets/...)"
pip install --upgrade pip
pip install mujoco==3.2.3 robosuite==1.4.1 gym==0.26.2 h5py==3.16.0 \
            torch==2.12.0 tyro==1.0.13 websockets==16.0 \
            opencv-python==4.11.0.86 pillow imageio \
            bddl cloudpickle easydict hydra-core einops matplotlib

# LIBERO package itself (editable)
echo "[3/3] install libero from local repo (editable)"
if [ ! -d "${LIBERO_REPO}/libero" ]; then
  echo "ERROR: LIBERO repo missing at ${LIBERO_REPO} — must tar-pipe from 4090d first"
  exit 1
fi
cd "${LIBERO_REPO}"
pip install -e .

# smoke
python -c "import libero; print(\"libero=\", libero.__version__)"
python -c "import mujoco, robosuite, gym, h5py, tyro, websockets; print(\"mujoco=\", mujoco.__version__, \"robosuite=\", robosuite.__version__)"
echo "=== LIBERO ENV ALL DONE $(date) ==="
