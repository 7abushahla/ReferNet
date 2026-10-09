# E9: Protocol C (IDNet published cross-type pair)

Two directions. The sweep tool has no direction axis, so pass it through `--extra`:

    python scripts/make_sweep.py --protocols C --backbones resnet18 --fusions single,difference,attention \
        --seeds 42,43,44 --name E9-WV-to-AZ --extra "--source WV --target AZ"
    python scripts/make_sweep.py --protocols C --backbones resnet18 --fusions single,difference,attention \
        --seeds 42,43,44 --name E9-AZ-to-WV --extra "--source AZ --target WV"

Compared with arXiv 2408.01690 v2, Table 14 (accuracy at threshold 0.5, per task). See docs/data-decision.md.
