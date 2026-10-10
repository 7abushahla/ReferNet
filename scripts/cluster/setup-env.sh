#!/bin/bash
# Build the refer-or-clear conda environment. Run ONCE on the LOGIN node; it needs network
# access, which the compute nodes do not have. Pattern from qcfs-speck/scripts/cluster/setup-env.sh.
#
#   ENV_PREFIX=/shared/$USER/conda/envs/refer-or-clear bash scripts/cluster/setup-env.sh
#
# The cluster driver is 570.86.15 (CUDA 12.8). Open-CD needs mmcv 2.x with compiled ops, and
# OpenMMLab publishes prebuilt mmcv wheels only for older torch versions. This script therefore
# pins the known-good Open-CD stack, torch 2.1.2 with CUDA 12.1 wheels, which a 12.8 driver runs.
# Stack verified against the opencd/__init__.py version asserts and the mmcv wheel index
# (evidence/code-audit-opencd-2026-10-10.md, A.4). If the wheel still fails on the cluster, build
# mmcv from source on a GPU node with MMCV_WITH_OPS=1.

set -euo pipefail

ENV_NAME="${ENV_NAME:-refer-or-clear}"
ENV_PREFIX="${ENV_PREFIX:-}"
CONDA_SH="${CONDA_SH:-/opt/miniconda/etc/profile.d/conda.sh}"
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.2}"
CU_INDEX="https://download.pytorch.org/whl/cu121"
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

home_avail=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -z "$ENV_PREFIX" && -n "${home_avail:-}" && "$home_avail" -lt 10 ]]; then
    echo "[setup-env] WARNING: \$HOME has only ${home_avail}G free; build on /shared:"
    echo "[setup-env]   ENV_PREFIX=/shared/\$USER/conda/envs/${ENV_NAME} bash scripts/cluster/setup-env.sh"
    sleep 10
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup-env] env '${ENV_NAME}' exists; reusing it"
elif [[ -n "$ENV_PREFIX" ]]; then
    [[ "$(basename "$ENV_PREFIX")" == "$ENV_NAME" ]] || { echo "[setup-env] FATAL: ENV_PREFIX must end in /$ENV_NAME" >&2; exit 1; }
    conda create -y -p "$ENV_PREFIX" python=3.10
    conda config --show envs_dirs | grep -qF -- "$(dirname "$ENV_PREFIX")" \
        || conda config --append envs_dirs "$(dirname "$ENV_PREFIX")"
else
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
[[ "$(basename "$CONDA_PREFIX")" == "$ENV_NAME" ]] || { echo "[setup-env] FATAL: wrong env active" >&2; exit 1; }

python -m pip install --upgrade pip
python -m pip install --index-url "$CU_INDEX" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
python -m pip install "mmcv==2.1.0" -f "$MMCV_INDEX"
python -m pip install -r requirements-train.txt
python -m pip install -e .

python - <<'PY'
import sys, torch
built = torch.version.cuda or ""
print(f"[setup-env] torch {torch.__version__} (CUDA build {built})")
if not built.startswith("12."):
    sys.exit(f"[setup-env] FATAL: torch is a CUDA-{built} build; this cluster needs a 12.x build")
PY

python scripts/check_imports.py
python -m pytest tests/ -q

cat <<EOF

[setup-env] done. Next, on the login node:
  bash scripts/cluster/stage-data.sh      # download, then build data/processed
  sbatch scripts/cluster/verify_gpu.sh    # 15-minute GPU-node check
EOF
