#!/bin/bash
# Download the SIDTD paper's released template checkpoints (spec diagnostic). Run on the LOGIN node (compute
# nodes are offline); unzips under data/raw/sidtd/paper_checkpoints/<model>/.
#   bash scripts/cluster/get_sidtd_ckpts.sh resnet50 [efficientnet-b3 ...]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D=data/raw/sidtd/paper_checkpoints
mkdir -p "$D"
for m in "$@"; do
    z="$D/${m}_trained_models.zip"
    wget -q -c -O "$z" "http://datasets.cvc.uab.es/SIDTD/balanced_templates_SIDTD/${m}_trained_models.zip"
    mkdir -p "$D/$m" && unzip -q -o "$z" -d "$D/$m"
    echo "$m: $(find "$D/$m" -name '*.pth' | wc -l) checkpoints"
done
