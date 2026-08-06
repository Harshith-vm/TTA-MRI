#!/usr/bin/env bash
# scripts/setup_env.sh — conda env for RTX A6000 (Ampere), Linux.
# Uses the same conda-forge recipe proven to work, with a CUDA pytorch build.
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n tta_patho -y -c conda-forge \
    python=3.10 pytorch-gpu torchvision pillow "numpy<2" scipy scikit-learn \
    pandas matplotlib seaborn scikit-image h5py tqdm statsmodels

conda activate tta_patho

pip install opencv-python-headless==4.9.0.80
pip install --no-deps grad-cam==1.5.0 ttach==0.0.3
pip install --no-deps albumentations==1.4.0 qudida==0.0.4
pip install pyyaml einops timm==0.9.16 tokenizers==0.19.1 transformers==4.40.0 \
    captum==0.7.0 scikit-posthocs pingouin huggingface-hub==0.23.0 accelerate \
    torchmetrics==1.3.0 wilds==2.0.0

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); \
assert torch.cuda.is_available(); print('bf16:', torch.cuda.is_bf16_supported())"
echo "Environment tta_patho ready."
