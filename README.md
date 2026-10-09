# Refer or Clear

Reference-conditioned detection of altered travel documents on designs never seen. Entry for the GDRFA Dubai Scientific Research Award, second cycle (Digital Innovation Award, artificial intelligence theme). Deadline 15 October 2026.

**Keep this repository private.** The award requires work that is original, unpublished, and not submitted elsewhere, and it screens submissions for originality. Do not make the repository public before the award decision.

## Layout

```
refer-or-clear/
├── paper/          award submission (LuaLaTeX, Overleaf-ready): build.sh, wordcount.sh, to_docx.sh
├── proposal/       full proposal and DESIGN.md (the plan every folder below implements)
├── evidence/       literature and evidence sweeps with source URLs
├── data/           raw/ interim/ processed/ (not committed; see data/README.md)
├── scripts/        download_datasets.sh, prepare_data.py, run_trufor_baseline.py
├── src/refer_or_clear/
│   ├── data/       records, rectify, references (design median, zone mask), forgeries, pairs, splits, dataset
│   ├── models/     build (ChangerEx via Open-CD), fusion (single, stacked, difference, attention), heads
│   ├── losses.py   metrics.py   train.py   evaluate.py
├── configs/        protocol_a.yaml, protocol_b.yaml (grid axes overridden on the command line)
├── experiments/    E1 to E9, one folder each, with the command and NOTES.md
├── results/        runs/ tables/ figures/ (small final outputs the paper cites)
├── slurm/          train_array.sh (48-task array: 4 backbones x 4 fusions x 3 seeds)
├── notebooks/      exploration only
└── tests/
```

Code files are placeholders: each states what it must do and where in `proposal/DESIGN.md` that is specified, and raises `NotImplementedError` until written. Datasets, checkpoints and logs never enter git.

## Workflow

```
bash scripts/download_datasets.sh          # about 10 GB into data/raw
python scripts/prepare_data.py             # records, rectification, references, pairs, splits
pip install -e .                           # the refer_or_clear package
sbatch --array=0-47 slurm/train_array.sh configs/protocol_a.yaml   # E1
sbatch --array=0-47 slurm/train_array.sh configs/protocol_b.yaml   # E2
python scripts/run_trufor_baseline.py      # Protocol B anchor
```

## Build

```
cd paper && bash build.sh        # needs LuaLaTeX, TeX Gyre Termes and Amiri (TeX Live)
cd proposal && bash build.sh     # pdfLaTeX + BibTeX
```

Notes in `proposal/README.md` and `proposal/DESIGN.md` refer to evidence files as `exa-results/<name>.md`; in this repository they are in `evidence/`.

## Authors

Hamza A. Abushahla, Ariel Justine N. Panopio, [researcher 3], Mohamed I. AlHajri. Department of Computer Science and Engineering, American University of Sharjah.
