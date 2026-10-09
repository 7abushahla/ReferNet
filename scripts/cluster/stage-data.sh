#!/bin/bash
# Download the datasets and build data/processed. Run on the LOGIN node; compute nodes are offline.
#
#   bash scripts/cluster/stage-data.sh
#
# Everything lands inside the repository on /shared (gitignored). Each training job then copies
# data/processed to node-local /scratch, because the pairs are read every epoch.

set -euo pipefail

ENV_NAME="${ENV_NAME:-refer-or-clear}"
CONDA_SH="${CONDA_SH:-/opt/miniconda/etc/profile.d/conda.sh}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"
[[ "$(basename "$CONDA_PREFIX")" == "$ENV_NAME" ]] || { echo "[stage-data] FATAL: wrong env" >&2; exit 1; }

bash scripts/cluster/download_datasets.sh
python scripts/prepare_data.py

# TODO: replace with the real counts once prepare_data.py exists (DESIGN.md 3.1, 3.5).
python - <<'PY'
from pathlib import Path
for f in ["data/processed/splits/protocol_a.csv", "data/processed/splits/protocol_b.csv", "data/processed/splits/protocol_c.csv"]:
    assert Path(f).is_file(), f"missing {f}"
print("[stage-data] splits present")
PY
