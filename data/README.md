# Data

Nothing in `raw/`, `interim/` or `processed/` is committed: the datasets are large and carry their own licences. Download with `scripts/download_datasets.sh`; every file is then rebuilt from code.

| Folder | Contents |
|---|---|
| `raw/midv2020/` | Optional (`WITH_MIDV=1`). Real photos and scans of the same 10 designs from `ftp://smartengines.com/midv-2020/dataset/`: `photo.tar` (4.0 GB), `scan_upright.tar` (1.15 GB). Not needed for Protocols A and B, because SIDTD already contains the MIDV-2020 templates. CC BY-SA 2.5. |
| `raw/sidtd/` | SIDTD ([github.com/Oriolrt/SIDTD_Dataset](https://github.com/Oriolrt/SIDTD_Dataset)), files from `http://datasets.cvc.uab.es/SIDTD/`: `templates.zip` (1.27 GB: the 1,000 MIDV-2020 genuine templates plus 1,222 generated forgeries, crop-and-replace and inpaint-and-rewrite, with annotations), `few_shot_split_templates.zip` (official 6/4 nationality split), `kfold_split_templates.zip`. SIDTD is MIDV-2020 plus generated forgeries, so it counts as one source. Repository licence CC BY 4.0; the MIDV-2020 images keep CC BY-SA 2.5. |
| `raw/fantasyid/` | FantasyID from Zenodo record 17063366: `FantasyID.tgz` (2.6 GB). CC BY 4.0. The validation part (record 17063494) needs a non-commercial agreement and is not used. |
| `raw/idnet/` | IDNet Zenodo release (CC0, 795 x 487 JPEG): `AZ.zip` 13.3 GB (record 13852757), `WV.zip` 31.6 GB (13854938); optional `GRC.zip` 8.9 GB, `ALB.zip` 25.6 GB (`WITH_IDNET_PASSPORTS=1`). Seven folders per design: positive and six fraud types, 5,979 images each, with field-level JSON. Reduced version of the paper's images. |
| `interim/` | Unified records (`records.parquet`), rectified pages, FantasyID layout annotations. |
| `processed/` | Design medians and zone masks per design, generated forgeries with exact masks, pair lists and the identity-grouped split files for Protocol A and Protocol B. |

Release: the derived data (layouts, medians, masks, splits) is released under the share-alike terms inherited from MIDV-2020 and SIDTD. No real traveller's document is used anywhere.

The decision on which datasets are used, and why, is in `docs/data-decision.md`.
