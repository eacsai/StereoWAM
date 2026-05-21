#!/usr/bin/env bash
# Install starVLA Python env on ksyun H100 dev container.
# Matches 4090d version pin: Python 3.10 / torch 2.6.0 / transformers 4.57 / flash-attn 2.7.4 / etc.
# NOTE: ksyun container default pypi mirror = http://pypi.ksyun.cn/simple/ (internal, fast).
#       download.pytorch.org is UNREACHABLE from container — so we do NOT pass --index-url.
#       torch 2.6.0 default wheel on PyPI ships with cu124 by default, which works on cu126 driver.
set -euo pipefail

REPO="/mnt/data/wangqiwei/wangqiwei/starVLA"
LOG="/mnt/data/wangqiwei/wangqiwei/install_env.log"
ENV_NAME="starvla"

source /opt/conda/etc/profile.d/conda.sh

# 1. create env (idempotent)
if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "[1/6] creating conda env ${ENV_NAME} (py3.10)"
  conda create -n "${ENV_NAME}" python=3.10 -y
else
  echo "[1/6] conda env ${ENV_NAME} already exists, reusing"
fi
conda activate "${ENV_NAME}"
python --version
which pip

# 2. torch 2.6.0 (ksyun pypi default, no pytorch.org index)
echo "[2/6] install torch 2.6.0 + torchvision 0.21.0 (via ksyun pypi)"
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0
python -c "import torch; print(\"torch=\", torch.__version__, \"cuda=\", torch.version.cuda, \"avail=\", torch.cuda.is_available(), \"devs=\", torch.cuda.device_count())"

# 3. requirements.txt (transformers 4.57 / accelerate 1.5 / deepspeed 0.16.9 / etc.)
echo "[3/6] install requirements.txt (via ksyun pypi)"
pip install -r "${REPO}/requirements.txt"

# 4. flash-attn + flash-linear-attention + causal_conv1d (per requirements.txt comment)
echo "[4/6] install flash-attn 2.7.4 + flash-linear-attention 0.3.2 + causal_conv1d 1.5.0"
pip install ninja  # required for flash-attn build
MAX_JOBS=16 pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install flash-linear-attention==0.3.2
pip install causal_conv1d==1.5.0.post8 --no-build-isolation

# 5. starVLA package itself (editable)
echo "[5/6] install starVLA in editable mode"
cd "${REPO}"
pip install -e .

# 6. smoke import test
echo "[6/6] smoke import test"
python -c "import torch; print(\"torch=\", torch.__version__, \"cuda=\", torch.version.cuda, \"avail=\", torch.cuda.is_available(), \"devs=\", torch.cuda.device_count())"
python -c "import transformers, accelerate, deepspeed, flash_attn; print(\"transformers=\", transformers.__version__, \"accelerate=\", accelerate.__version__, \"deepspeed=\", deepspeed.__version__, \"flash_attn=\", flash_attn.__version__)"
python -c "import fla; print(\"flash_linear_attention OK\")" || echo "(fla import failed, but maybe non-blocking)"
python -c "from starVLA.model.framework import build_framework; print(\"build_framework import OK\")"

echo "=== ALL DONE $(date) ==="
