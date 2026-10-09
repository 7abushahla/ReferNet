"""Build everything under data/interim and data/processed from data/raw.

Steps (DESIGN.md section 3):
  1. extract archives and build unified records       -> refer_or_clear.data.records
  2. rectify every page onto its design canvas        -> refer_or_clear.data.rectify
  3. design medians and zone masks per design         -> refer_or_clear.data.references
  4. extra forgeries with the SIDTD generator          -> refer_or_clear.data.forgeries
  5. pairs and identity-grouped splits (A and B)      -> refer_or_clear.data.pairs, .splits
"""


def main():
    raise NotImplementedError("Day 1 to 2 of the plan; see DESIGN.md section 7.")


if __name__ == "__main__":
    main()
