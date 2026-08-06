#!/bin/bash
# scripts/run_all.sh — full pipeline, safe to re-run (skips completed steps).
# All artifacts are written beside the code on the same disk (see config.py).
set -e

# Run from the repo root regardless of where this is invoked from.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
# A6000 / Ampere performance knobs
export TORCH_CUDNN_V8_API_ENABLED=1
export NVIDIA_TF32_OVERRIDE=1            # allow TF32 matmuls
export OMP_NUM_THREADS=8

echo "================================================================"
echo "  TTA × Explainability Pipeline — Full Run"
echo "  Repo root: $(pwd)"
echo "================================================================"

MODELS="resnet50 efficientnet_b3 swin_t vit_b16 phikon_v2 h_optimus_0 prov_gigapath"
SEEDS="17 31 53 97 211"
CNN_EXTRA_SEEDS="307 401"

echo "[1/5] Downloading / verifying datasets..."
python data/download_all.py || true
python scripts/verify_downloads.py || true

echo "[2/5] Source-domain training (PCam->Camelyon17 and NCT-CRC-100K->7K)..."
for DATASET in camelyon17 nct_crc; do
  for MODEL in $MODELS; do
    SEEDLIST="$SEEDS"
    if [ "$MODEL" = "resnet50" ] || [ "$MODEL" = "efficientnet_b3" ]; then
      SEEDLIST="$SEEDS $CNN_EXTRA_SEEDS"
    fi
    for SEED in $SEEDLIST; do
      echo "  train: $DATASET | $MODEL | seed=$SEED"
      python training/train_source.py --model "$MODEL" --seed "$SEED" --dataset "$DATASET"
    done
  done
done

echo "[3/5] TTA evaluation (resumable; multi-XAI, faithfulness)..."
python run_evaluation.py

echo "[4/5] Statistical analysis (LMM, paired Wilcoxon+FDR, Friedman/Nemenyi)..."
python analysis/run_all_analysis.py

echo "[5/5] Publication figures..."
python figures/generate_all_figures.py

echo "================================================================"
echo "  Complete. See results/ , results/aggregated/ , figures/output/"
echo "================================================================"
python run_evaluation.py --status
