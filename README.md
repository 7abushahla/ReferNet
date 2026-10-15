# Refer or Clear

Reference-conditioned detection of altered travel documents on designs never seen. Entry for the GDRFA Dubai Scientific Research Award, second cycle (Digital Innovation Award, artificial intelligence theme). 

## Layout

```
refer-or-clear/
├── proposal/       research proposal (LuaLaTeX); the award manuscript lives in the separate GDRFA-Award-Paper repository
├── proposal/       full proposal and DESIGN.md (the plan every folder below implements)
├── evidence/       literature and evidence sweeps with source URLs
├── data/           raw/ interim/ processed/ (not committed; see data/README.md)
├── scripts/
│   ├── make_sweep.py       generates SLURM launchers for a training grid (never hand-write one)
│   ├── cluster/            setup-env.sh, stage-data.sh, download_datasets.sh, verify_gpu.sh
│   ├── prepare_data.py     records, rectification, references, pairs, splits
│   ├── run_trufor_baseline.py   Protocol B anchor
│   └── check_imports.py
├── src/refer_or_clear/
│   ├── data/       records, rectify, references (design median, zone mask), forgeries, pairs, splits, dataset
│   ├── models/     build (ChangerEx via Open-CD), fusion (single, stacked, difference, attention), heads
│   ├── losses.py   metrics.py   train.py   evaluate.py
├── configs/        protocol_a.yaml, protocol_b.yaml (grid axes overridden on the command line)
├── sweeps/         generated launchers, one folder per sweep, committed with their results
├── experiments/    E1 to E9, one folder each, with notes
├── results/        runs/ tables/ figures/ (small final outputs the paper cites)
├── docs/cluster.md the AWS SLURM cluster: setup, sweeps, storage, budget
├── notebooks/      exploration only
└── tests/
```

Code files are placeholders: each states what it must do and where in `proposal/DESIGN.md` that is specified, and raises `NotImplementedError` until written. The cluster tooling (`make_sweep.py`, `scripts/cluster/`) follows the qcfs-speck repository and is ready to use. Datasets, checkpoints and logs never enter git.

## Workflow (cluster; details in docs/cluster.md)

```
ENV_PREFIX=/shared/$USER/conda/envs/refer-or-clear bash scripts/cluster/setup-env.sh   # login node, once
bash scripts/cluster/stage-data.sh                     # login node: download about 10 GB, build data/processed
sbatch scripts/cluster/verify_gpu.sh                   # 15-minute GPU check
python scripts/make_sweep.py --protocols A --backbones resnet18,resnet50,resnest50,mit_b1 \
    --fusions single,stacked,difference,attention --name E1-protocolA
bash sweeps/E1-protocolA/submit_all.sh                 # from the repository root
```

## Build

```
cd paper && bash build.sh        # needs LuaLaTeX, TeX Gyre Termes and Amiri (TeX Live)
cd proposal && bash build.sh     # pdfLaTeX + BibTeX
```

Notes in `proposal/README.md` and `proposal/DESIGN.md` refer to evidence files as `exa-results/<name>.md`; in this repository they are in `evidence/`.

## Demo

A live demo runs the paper's fold-0 deployment model in the browser: https://7abushahla.github.io/ReferNet/ (source in
`site/`). Its data are rebuilt with `python demo/export_onnx.py`, `python demo/dump_inputs.py` (cluster, GPU) and
`python demo/build_site_data.py`; `/verify.html` checks every browser score against PyTorch.

## Authors

Hamza A. Abushahla, Ariel Justine N. Panopio, [researcher 3], Mohamed I. AlHajri. Department of Computer Science and Engineering, American University of Sharjah.
