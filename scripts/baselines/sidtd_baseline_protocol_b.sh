#!/bin/bash
# Usage: sidtd_baseline_protocol_b.sh GROUP MODEL [EPOCHS]   (submits one GPU job; always passes --time)
G=$1; M=$2; EP=${3:-100}
DIAG=/shared/b00090279/diag-align; REPO=/shared/b00090279/refer-or-clear
RUN=$(ls -d $REPO/results/runs/protocol-b-current/protocol-b-group${G}-* | head -1)
NAME=bl-g${G}-${M%%-*}
sbatch --account=acc-mialhajri --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=06:00:00 \
  --job-name=$NAME --output=$DIAG/logs/%x-%j.out --wrap="cd $DIAG && /shared/b00090279/conda/envs/refer-or-clear/bin/python -I baselines/sidtd_baseline_protocol_b.py --run-dir $RUN --raw-root $REPO/data/raw --model $M --epochs $EP --out $DIAG/baselines-b/group${G}/${M}"
