"""Train one configuration: python -m refer_or_clear.train --config configs/protocol_a.yaml --seed 42

AdamW, lr 1e-4 backbone and 1e-3 new blocks, cosine schedule, batch 8 at 512 x 800, 40 epochs,
mixed precision, early stopping on identity-disjoint validation AUC. Seeds 42, 43, 44.
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
