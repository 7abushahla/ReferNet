# Data

Nothing in `raw/`, `interim/` or `processed/` is committed: the datasets are large and carry their own licences. Download with `scripts/download_datasets.sh`; every file is then rebuilt from code.

| Folder | Contents |
|---|---|
| `raw/midv2020/` | MIDV-2020 from `ftp://smartengines.com/midv-2020/dataset/`: `templates.tar` (0.86 GB), `photo.tar` (4.0 GB), `scan_upright.tar` (1.15 GB). Clips optional. CC BY-SA 2.5. |
| `raw/sidtd/` | SIDTD from `http://datasets.cvc.uab.es/SIDTD/`: `templates.zip` (1.27 GB) and the split files `split_shot.zip`, `split_kfold.zip`, `split_normal.zip`. CC BY-SA. |
| `raw/fantasyid/` | FantasyID from Zenodo record 17063366: `FantasyID.tgz` (2.6 GB). CC BY 4.0. The validation part (record 17063494) needs a non-commercial agreement and is not used. |
| `raw/idnet/` | Optional. IDNet-2025 from Hugging Face `cactuslab/IDNet-2025`, two to four EU country archives only. |
| `interim/` | Unified records (`records.parquet`), rectified pages, FantasyID layout annotations. |
| `processed/` | Design medians and zone masks per design, generated forgeries with exact masks, pair lists and the identity-grouped split files for Protocol A and Protocol B. |

Release: the derived data (layouts, medians, masks, splits) is released under the share-alike terms inherited from MIDV-2020 and SIDTD. No real traveller's document is used anywhere.
