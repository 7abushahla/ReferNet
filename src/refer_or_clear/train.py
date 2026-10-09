"""Train one configuration (called by the generated sweep jobs, scripts/make_sweep.py):

    python -m refer_or_clear.train --config configs/protocol_a.yaml --backbone resnet18 \
        --fusion difference --reference design_median --transfer full --seed 42 \
        --data-root $ROC_DATA --run-dir results/runs/<tag>

Must write <run-dir>/metrics.json last, only on success: the sweep skips arms whose metrics.json exists.

AdamW, lr 1e-4 backbone and 1e-3 new blocks, cosine schedule, batch 8 at 512 x 800, 40 epochs,
mixed precision, early stopping on identity-disjoint validation AUC. Seeds 42, 43, 44.
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
