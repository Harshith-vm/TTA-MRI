# data/nct_crc_dataset.py
"""
NCT-CRC-HE colorectal histology, 9 tissue classes.
  - NCT-CRC-HE-100K : source/train (100k patches, Macenko-normalised cohort)
  - CRC-VAL-HE-7K   : OOD test (different patients/scanners -> natural shift)

Provides a multi-class benchmark alongside the binary Camelyon17, so the paper
can claim generality across tissue types and class cardinality.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _index_split(root: Path):
    """Return list of (path, label) for every .tif under class subfolders."""
    samples = []
    for c in CLASSES:
        d = root / c
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.tif")):
            samples.append((p, CLASS_TO_IDX[c]))
    return samples


class NCTCRCDataset(Dataset):
    """
    split='train' -> NCT-CRC-HE-100K ; split='val7k' -> CRC-VAL-HE-7K (OOD).
    `root` defaults to <repo>/datasets/nct_crc.
    """
    def __init__(self, split="val7k", root=None, transform=None):
        from config import DATASETS_ROOT
        root = Path(root) if root else DATASETS_ROOT / "nct_crc"
        sub = "NCT-CRC-HE-100K" if split == "train" else "CRC-VAL-HE-7K"
        self.root = root / sub
        self.transform = transform
        self.samples = _index_split(self.root)
        if not self.samples:
            raise FileNotFoundError(f"No NCT-CRC images under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label
