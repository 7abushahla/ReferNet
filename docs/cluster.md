# Running on the SLURM cluster

Same AWS SLURM cluster and conventions as the qcfs-speck repository (`docs/cluster.md` there, measured 28 July 2026): NVIDIA A10G GPUs with 24 GB, driver 570.86.15 (CUDA 12.8).

## Layout

| Thing | Where |
|---|---|
| Repository | anywhere on `/shared`; everything lives inside it |
| Datasets | `<repo>/data/raw`, `<repo>/data/interim`, `<repo>/data/processed` (gitignored), built by `scripts/cluster/stage-data.sh` |
| Per-job data | `data/processed` copied to `/scratch/$USER/roc-data` at job start (`ROC_DATA`) |
| Results | `results/runs/<tag>/metrics.json` (committed); checkpoints in `checkpoints/` (not committed) |
| SLURM logs | `logs/slurm/<job-name>-<jobid>.out` (gitignored) |
| Conda | `/opt/miniconda/etc/profile.d/conda.sh`, env `refer-or-clear` |
| SLURM | `--account=acc-mialhajri --partition=gpu --gres=gpu:1` |
| Caps per job | `--cpus-per-task=4`, `--mem=8G` |

## First-time setup (login node, which has network)

```bash
git clone git@github.com:7abushahla/refer-or-clear.git /shared/$USER/refer-or-clear
cd /shared/$USER/refer-or-clear
ENV_PREFIX=/shared/$USER/conda/envs/refer-or-clear bash scripts/cluster/setup-env.sh   # torch 2.1.2 cu121, mmcv 2.1, Open-CD, tests
bash scripts/cluster/stage-data.sh      # download about 10 GB, build data/processed
sbatch scripts/cluster/verify_gpu.sh    # 15 minutes; then read logs/slurm/roc-verify-<jobid>.out
```

`$HOME` is nearly full on this cluster, so the env goes on `/shared`. The torch pin differs from qcfs-speck (2.10 cu128) because Open-CD needs mmcv with compiled ops and OpenMMLab's prebuilt wheels stop at torch 2.1; the 12.8 driver runs cu121 builds. Confirm on day 3.

## Sweeps

Launchers are generated, never hand-written. Generate, inspect, then submit from the repository root.

```bash
# E1: Protocol A main grid, every arm on its own GPU: 4 backbones x 4 fusions x 3 seeds = 48 jobs
python scripts/make_sweep.py --protocols A --backbones resnet18,resnet50,resnest50,mit_b1 \
    --fusions single,stacked,difference,attention --seeds 42,43,44 --name E1-protocolA
bash sweeps/E1-protocolA/submit_all.sh

# E2: the same grid under Protocol B
python scripts/make_sweep.py --protocols B --backbones resnet18,resnet50,resnest50,mit_b1 \
    --fusions single,stacked,difference,attention --name E2-protocolB

# E4 and E5: ablations on one backbone, seeds back to back per job
python scripts/make_sweep.py --protocols A --backbones resnet18 --fusions difference \
    --transfers frozen,partial,full,scratch --group-size 3 --time 30:00:00 --name E4-transfer
python scripts/make_sweep.py --protocols A,B --backbones resnet18 --fusions difference,attention \
    --references design_median,masked_exemplar --name E5-reference

squeue -u $USER
tail -f logs/slurm/<job-name>-<jobid>.out
```

Jobs are resumable at the arm level: an arm whose `results/runs/<tag>/metrics.json` exists is skipped. A failing arm does not abort the others; the job prints a summary and exits nonzero. Each job checks, before training, that it runs from the repository root, that `data/processed` exists, that staging to `/scratch` worked, and that a real cuDNN convolution runs on the allocated GPU.

`CUDA_VISIBLE_DEVICES` is set by SLURM; do not set it in a job script.

## Storage

| Path | Type | Read | Use |
|---|---|---|---|
| `/shared` | NFS | about 100 MB/s | repository, data and results at rest |
| `/scratch` | node-local NVMe | about 2.4 GB/s | `TMPDIR` and the per-job data copy |
| `/home` | NFS | | nearly full; keep conda envs and data out |
| `/tmp` | root EBS volume | 11.7 MB/s | avoid; jobs redirect `TMPDIR` to `/scratch` |

Unlike CIFAR-10 in qcfs-speck, the pairs here are image files read every epoch, so each job copies `data/processed` to `/scratch` once at start.

## Budget

The E1 and E2 grids are 96 single-GPU jobs; at a few hours each at 512 x 800 that is roughly 300 GPU-hours. If the queue is short, cut to two backbones (`--backbones resnet18,resnet50`).
