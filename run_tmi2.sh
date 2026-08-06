#!/usr/bin/env bash

set -Eeuo pipefail

########################################
# Repository
########################################

cd "$(dirname "$0")"
export MRI_ROOT="$(pwd)"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=8

echo "======================================="
echo "Repository : $MRI_ROOT"
echo "======================================="

########################################
# Write Permission
########################################

touch "$MRI_ROOT/.wtest" 2>/dev/null || {
    echo "Repository is read-only."
    exit 1
}
rm -f "$MRI_ROOT/.wtest"

########################################
# Locate Conda
########################################

CONDA_BASE=""

for c in \
"$HOME/miniconda3" \
"$HOME/anaconda3" \
"$HOME/miniforge3" \
"/opt/conda" \
"/opt/miniconda3"
do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$c"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    CONDA_BASE=$(conda info --base)
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

########################################
# Install Mamba if missing
########################################

if ! command -v mamba >/dev/null 2>&1; then
    echo "Installing mamba..."
    conda install -n base -c conda-forge -y mamba
fi

########################################
# Create Environment
########################################

if ! conda env list | grep -q "^tta_patho"; then

    echo
    echo "Creating environment..."
    echo

    mamba create -y \
        -n tta_patho \
        python=3.10 \
        pytorch=2.5.* \
        torchvision=0.20.* \
        pytorch-cuda=12.4 \
        numpy=1.26 \
        scipy \
        pandas \
        matplotlib \
        seaborn \
        scikit-image \
        scikit-learn \
        statsmodels \
        pillow \
        h5py \
        tqdm \
        pip \
        -c pytorch \
        -c nvidia \
        -c conda-forge

fi

conda activate tta_patho

########################################
# Install Python Packages
########################################

python -m pip install --upgrade pip

pip install \
opencv-python-headless==4.9.0.80 \
grad-cam==1.5.0 \
ttach==0.0.3 \
albumentations==1.4.0 \
qudida==0.0.4 \
pyyaml \
einops \
timm==0.9.16 \
tokenizers==0.19.1 \
transformers==4.40.0 \
captum==0.7.0 \
scikit-posthocs \
pingouin \
huggingface-hub==0.23.0 \
accelerate \
torchmetrics==1.3.0 \
wilds==2.0.0

########################################
# Verify CUDA
########################################

python << EOF

import torch

print("="*60)
print("Torch Version :", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"

print("GPU:", torch.cuda.get_device_name(0))
print("="*60)

EOF

########################################
# Variables
########################################

MODELS="resnet50 efficientnet_b3 swin_t vit_b16 phikon_v2"

SEEDS="17 31 53 97 211"

CNN_EXTRA="307 401"

########################################
# Training
########################################

echo
echo "================ TRAINING ================"
echo

for DS in camelyon17 nct_crc
do

    for MODEL in $MODELS
    do

        CUR_SEEDS="$SEEDS"

        if [[ "$MODEL" == "resnet50" || "$MODEL" == "efficientnet_b3" ]]; then
            CUR_SEEDS="$CUR_SEEDS $CNN_EXTRA"
        fi

        for SEED in $CUR_SEEDS
        do

            echo
            echo "Dataset : $DS"
            echo "Model   : $MODEL"
            echo "Seed    : $SEED"
            echo

            python training/train_source.py \
                --dataset "$DS" \
                --model "$MODEL" \
                --seed "$SEED"

        done

    done

done

########################################
# Evaluation
########################################

python reset_eval.py
python run_evaluation.py

########################################
# Ablation
########################################

for B in 16 32 64 128
do

python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model resnet50 --method tent --eval_bs $B --tag bs$B

python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model vit_b16 --method cotta --eval_bs $B --tag bs$B

done

python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model resnet50 --method cotta --reset --tag episodic

python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model vit_b16 --method tent --reset --tag episodic

for STEP in 1 2 4 8
do

python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model resnet50 --method tent --steps $STEP --tag steps$STEP

python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model vit_b16 --method sar --steps $STEP --tag steps$STEP

done

########################################
# Analysis
########################################

python analysis/run_all_analysis.py
python postprocess/make_outputs.py
python figures/generate_all_figures.py

echo
echo "========== COMPLETE =========="

python run_evaluation.py --status
