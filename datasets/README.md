# datasets/

This folder holds all raw datasets used in the **TTA Explainability Pipeline for Histopathology**.

| Dataset | License | Size | Task | Source |
|---|---|---|---|---|
| **PatchCamelyon (PCam)** | CC0 | ~8 GB | Binary: normal vs. tumour | [Zenodo #2546921](https://zenodo.org/record/2546921) |
| **Camelyon17-WILDS** | CC0 | ~10 GB | Binary across 5 hospitals | [WILDS benchmark](https://wilds.stanford.edu/) |
| **NCT-CRC-HE-100K** | CC0 | ~3.8 GB | 9-class CRC tissue | [Zenodo #1214456](https://zenodo.org/record/1214456) |
| **CRC-VAL-HE-7K** | CC0 | ~0.5 GB | 9-class CRC tissue (val) | [Zenodo #1214456](https://zenodo.org/record/1214456) |

## Directory layout after download

```
datasets/
├── pcam/
│   ├── camelyonpatch_level_2_split_train_x.h5
│   ├── camelyonpatch_level_2_split_train_y.h5
│   ├── camelyonpatch_level_2_split_valid_x.h5
│   ├── camelyonpatch_level_2_split_valid_y.h5
│   ├── camelyonpatch_level_2_split_test_x.h5
│   └── camelyonpatch_level_2_split_test_y.h5
├── wilds/
│   └── camelyon17_v2.0/
│       └── ... (patches + metadata)
├── nct_crc/
│   ├── NCT-CRC-HE-100K/
│   │   └── <class>/*.tif  (100,000 patches)
│   └── CRC-VAL-HE-7K/
│       └── <class>/*.tif  (7,180 patches)
└── download_datasets.py   ← this script
```

## How to download

```powershell
# Download all datasets (resume-safe — re-run if interrupted)
python datasets/download_datasets.py

# Skip individual datasets
python datasets/download_datasets.py --skip-pcam
python datasets/download_datasets.py --skip-wilds
python datasets/download_datasets.py --skip-nct
```

## Notes

- All downloads are **resume-safe** — re-running the script skips files already on disk.
- PCam HDF5 `.gz` files are decompressed automatically; the `.gz` is deleted after decompression to save disk space.
- NCT-CRC ZIP files are extracted and then deleted to save disk space.
- Camelyon17-WILDS is downloaded via the `wilds` Python package (auto-installed if missing).
- No registration or login is required for any dataset.
