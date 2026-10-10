#!/bin/bash
# Run Protocol arms back to back on a local GPU (the lab 3080), same skip rule as the cluster jobs.
#   bash scripts/local_queue.sh <queue-file>     # lines: protocol arch fusion reference init seed
# Logs to results/runs/<tag>.log; an arm whose metrics.json exists is skipped.
set -u
QUEUE="$1"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
declare -A CFG=([A]=configs/protocol_a.yaml [B]=configs/protocol_b.yaml [C]=configs/protocol_c.yaml [D]=configs/protocol_d.yaml)
# Protocols A and B need --fold (train.py); this lab-GPU queue does not pass it, so use the cluster sweeps for them.
declare -A SHORT=([siamese_r18]=r18 [changerex_r18]=cex)
while read -r proto arch fusion ref init seed; do
    [[ -z "$proto" || "$proto" == \#* ]] && continue
    tag="${proto}-${SHORT[$arch]}-${fusion}-${ref}-${init}-s${seed}"
    run="results/runs/$tag"
    if [[ -f "$run/metrics.json" ]]; then echo "[skip] $tag"; continue; fi
    echo "[run] $tag $(date '+%F %T')"
    python -u -m refer_or_clear.train --config "${CFG[$proto]}" --arch "$arch" --fusion "$fusion" \
        --reference "$ref" --init "$init" --seed "$seed" --data-root data/processed --run-dir "$run" \
        --amp --workers 6 > "$run.log" 2>&1 && echo "[done] $tag $(date '+%F %T')" || echo "[FAIL] $tag"
done < "$QUEUE"
