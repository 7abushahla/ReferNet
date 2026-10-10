#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=roc-verify
#SBATCH --output=logs/slurm/%x-%j.out
#
# GPU-node check with the same SBATCH resources as a training job. Submit from the repo root:
#
#   sbatch scripts/cluster/verify_gpu.sh
#   cat logs/slurm/roc-verify-<jobid>.out
#
# Runs: imports, a cuDNN convolution, one forward and backward pass per arm at the training batch
# (peak memory printed), then the two-epoch GPU smoke test on 64 pairs (writes results/runs/smoke-gpu).

set -uo pipefail

ENV="${ENV:-refer-or-clear}"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$ENV"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -d src/refer_or_clear ]] || { echo "[FATAL] submit from the repo root, not $PWD"; exit 1; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="/scratch/$USER/tmp"; mkdir -p "$TMPDIR"

echo "node      : $(hostname)   job ${SLURM_JOB_ID:-none}"
echo "gpu       : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo none)"
echo "cpus      : $(nproc) visible / ${SLURM_CPUS_PER_TASK:-?} allocated"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python scripts/check_imports.py || exit 1
python - <<'PYCHK' || exit 1
import time, torch, sys
sys.path.insert(0, "src")
from refer_or_clear.models import build_model, count_params
from refer_or_clear.losses import referral_loss
assert torch.cuda.is_available(), "CUDA not available"
print(f"[ok] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}")
for arch, init in (("siamese_r18", "imagenet"), ("changerex_r18", "levir")):
    try:
        m = build_model(arch, "difference", init).cuda().train()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {arch}: {e}"); continue
    x = torch.randn(8, 8, 448, 704, device="cuda"); y = torch.randint(0, 2, (8,), device="cuda").float()
    mask = torch.zeros(8, 1, 448, 704, device="cuda"); valid = torch.ones_like(mask)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        out = m(x); loss, parts = referral_loss(out["logit"], out["map"], y, mask, valid)
    loss.backward(); torch.cuda.synchronize()
    print(f"[ok] {arch}: {count_params(m)/1e6:.1f} M params, batch 8 at 448x704 fwd+bwd {time.time()-t0:.2f} s, "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB, loss {loss.item():.3f}")
    del m, x, out, loss; torch.cuda.empty_cache()
PYCHK
DEVICE=cuda LIMIT=64 EPOCHS=2 BATCH=8 CROP="448 704" RUN_DIR=results/runs/smoke-gpu bash scripts/smoke_test.sh || exit 1
echo "[ok] verify_gpu passed"
