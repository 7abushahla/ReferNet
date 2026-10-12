#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --job-name=getter-profile
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --nodelist=gpu-dy-g5-0-17
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/b00090279/conda/envs/refer-or-clear
unset PYTHONPATH PYTHONHOME ROC_ANCHOR_GPU_INTEGRATION ROC_DIRECT_REFERENCE_GPU_INTEGRATION
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python_bin=/shared/b00090279/conda/envs/refer-or-clear/bin/python
root=/shared/b00090279/refer-or-clear
bundle="$root/results/source/support-c2617062fdb352fb7370e34c6768313409e9cb1d"
stage=/scratch/b00090279/direct-reference-throughput-38088
run="$root/results/diagnostics/direct-reference-v1/direct-reference-throughput-38088"
request_sha=36247e1ac7ef13dd9944dd5fd6aa6d191765ce28e4e2362648cc69171a6a8ba8
profiler="$root/results/source/getter-profiler-2844ccf/profile_direct_reference_getter.py"
printf '%s  %s\n' d9a8ce0e648f7723a186e504daba5d4d06fbcd4f1735671c1e827432f36dc043 "$profiler" | sha256sum -c -
cd "$root"
if [[ ! -e "$stage" && ! -L "$stage" ]]; then
    "$python_bin" -I -B "$bundle/code/scripts/direct_reference_job.py" prepare \
        --source-dir "$bundle/code" \
        --source-sha256 c1c59cefd2dce9a1815b235f81f763eb8781667853a8d80bf36666a71074f815 \
        --request "$bundle/throughput-request.json" --request-sha256 "$request_sha" \
        --stage-dir "$stage" \
        --run-dir "$root/results/diagnostics/direct-reference-v1/getter-restaging-$SLURM_JOB_ID"
fi
export TMPDIR="$stage/tmp"
export ROC_LEVIR_CKPT="$stage/inputs/weights.pth"
export ROC_LEVIR_SHA256="$("$python_bin" -I -B -c 'import hashlib,json,sys; payload=open(sys.argv[1],"rb").read(); assert hashlib.sha256(payload).hexdigest()==sys.argv[2]; print(json.loads(payload)["weights"]["sha256"])' "$stage/inputs/request.json" "$request_sha")"
"$python_bin" -I -B "$profiler" \
    --source-dir "$stage/code" --request "$stage/inputs/request.json" \
    --request-sha256 "$request_sha" --run-dir "$run" \
    --report "$run/throughput/report.json" \
    --report-sha256 70dda430267086ceb884013e2a70a3e0aa88f5cf78e419b08b597f785088766f \
    --output-root "$root/results/diagnostics/direct-reference-v1/getter-profile-$SLURM_JOB_ID"
