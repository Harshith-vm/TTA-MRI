# data/pcam_dataset.py
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class PCamDataset(Dataset):
    def __init__(self, split='train', root_dir=None, transform=None):
        if root_dir is None:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            import config
            root_dir = config.DATASETS_ROOT / "pcam"   # data lives in datasets/pcam
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Determine files based on split
        self.x_path = self.root_dir / f"camelyonpatch_level_2_split_{split}_x.h5"
        self.y_path = self.root_dir / f"camelyonpatch_level_2_split_{split}_y.h5"
        
        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(f"PCam dataset files for {split} not found at {self.root_dir}. Run download_all.py first.")
            
        # Keep open handles. Note: not multiprocessing safe out of the box if using num_workers > 0 unless handle opened per worker
        # but for simplicity, we open it here. Real robust implementation might use a worker_init_fn
        self._x_file = None
        self._y_file = None
        
        # Just to get length:
        with h5py.File(self.x_path, 'r') as xf:
            self.length = xf['x'].shape[0]

    def _init_files(self):
        if self._x_file is None:
            self._x_file = h5py.File(self.x_path, 'r')
            self._y_file = h5py.File(self.y_path, 'r')
            self.x_data = self._x_file['x']
            self.y_data = self._y_file['y']

    # --- pickle safety (Windows uses 'spawn', which pickles the dataset) ------
    def __getstate__(self):
        """Never pickle the open h5py handles; workers reopen lazily."""
        state = self.__dict__.copy()
        state['_x_file'] = None
        state['_y_file'] = None
        state.pop('x_data', None)
        state.pop('y_data', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._x_file = None
        self._y_file = None

    def get_all_labels(self):
        """Read every label directly (no image decode) for the balanced sampler."""
        with h5py.File(self.y_path, 'r') as yf:
            return np.asarray(yf['y']).reshape(-1).astype(int)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_files()
        
        # PCam HDF5 returns (96, 96, 3) image
        img = self.x_data[idx]
        
        # PCam labels are shaped like (1,1,1) in some versions, flatten and take first
        label = int(self.y_data[idx].flatten()[0])
        
        if self.transform:
            # Albumentations expects a numpy array, which img already is (H, W, C)
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return img, label

    def __del__(self):
        try:
            if self._x_file is not None:
                self._x_file.close()
                self._y_file.close()
        except:
            pass
