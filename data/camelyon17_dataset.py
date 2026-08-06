# data/camelyon17_dataset.py
"""
Reads Camelyon17 patches DIRECTLY from datasets/wilds — no `wilds` package and
no special folder layout required. This matches the data exactly as it sits on
the drive:

    datasets/wilds/metadata.csv
    datasets/wilds/patches/patient_<PPP>_node_<N>/patch_patient_<PPP>_node_<N>_x_<X>_y_<Y>.png

metadata columns: ,patient,node,x_coord,y_coord,tumor,slide,center,split
We filter to a single hospital `center` (the TTA domain) and use `tumor` as the
binary label.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class Camelyon17CenterDataset(Dataset):
    def __init__(self, center_id, root_dir=None, transform=None):
        self.center_id = int(center_id)
        self.transform = transform
        root = Path(root_dir) if root_dir else (config.DATASETS_ROOT / "wilds")
        self.patches_dir = root / "patches"
        meta_path = root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Camelyon17 metadata.csv not found at {meta_path}")
        if not self.patches_dir.is_dir():
            raise FileNotFoundError(f"Camelyon17 patches/ not found at {self.patches_dir}")
        df = pd.read_csv(meta_path)
        self.df = df[df["center"] == self.center_id].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No Camelyon17 patches for center {self.center_id}")

    def __len__(self):
        return len(self.df)

    def _patch_path(self, row):
        p = f"{int(row['patient']):03d}"
        n = int(row["node"])
        x = int(row["x_coord"]); y = int(row["y_coord"])
        folder = f"patient_{p}_node_{n}"
        fname = f"patch_patient_{p}_node_{n}_x_{x}_y_{y}.png"
        return self.patches_dir / folder / fname

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(self._patch_path(row)).convert("RGB"))
        label = int(row["tumor"])
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

