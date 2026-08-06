# models/transforms.py
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# ImageNet stats for CNN/ViT models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Pathology-optimized stats from PCam training set
PCAM_MEAN = [0.7008, 0.5384, 0.6916]
PCAM_STD  = [0.2350, 0.2774, 0.2128]

def get_transforms(model_name: str, split: str = 'train'):
    """
    Returns albumentations Compose for train/val/test.
    NOTE: Foundation models use their own preprocessing conventions.
    """
    is_foundation = model_name in ('phikon_v2', 'h_optimus_0', 'prov_gigapath')
    is_vit = model_name in ('vit_b16',)
    is_swin = model_name in ('swin_t',)

    # All non-foundation models use 224×224 input
    target_size = 224

    # Foundation models: use pathology-tuned normalization
    mean = PCAM_MEAN if is_foundation else IMAGENET_MEAN
    std  = PCAM_STD  if is_foundation else IMAGENET_STD

    if split == 'train':
        return A.Compose([
            A.Resize(target_size, target_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Conservative H&E color augmentation — stain variation is biologically real
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(target_size, target_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
