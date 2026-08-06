#!/usr/bin/env bash
# ============================================================================
# run_tmi.sh — ONE script to run the whole corrected pipeline on Linux + A6000.
# Place this at the repo root (beside config.py) and run:  bash run_tmi.sh
#
# It: (0) builds the conda env if missing, (1) trains any missing checkpoints
# (skips existing), (2) resets old buggy eval results, (3) fp32 re-run of the
# full grid with all fixes, (4) ablations, (5) analysis + figures + tables.
# Fully resumable — re-run it any time; completed work is skipped.
#
# Post-crash hardening (see run_evaluation.py): each of the ~3000 evaluation
# combos runs in its OWN subprocess so a leak in torch/HF/timm cannot compound.
# Failures never stop the sweep — every step keeps going and the SQLite
# registry lets you re-run this script to catch anything the crash missed.
# ============================================================================
set -uo pipefail                     # NOT -e: one failing combo must not
                                     # abort the remaining 2999.
cd "$(dirname "$0")"                     # repo root
export MRI_ROOT="$(pwd)"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Cut VRAM fragmentation across variable batch sizes; cut RSS bloat in the
# long-lived parent process.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_TRIM_THRESHOLD_=131072
export TOKENIZERS_PARALLELISM=false

echo "Repo root: $MRI_ROOT"
if [ -f "/home/ise-vc/myenv/bin/activate" ]; then
  source "/home/ise-vc/myenv/bin/activate"
elif [ -f "/home/atr_ir/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/home/atr_ir/miniconda3/etc/profile.d/conda.sh"
  if ! conda env list | grep -qE '^\s*tta_patho\s'; then
    echo "=== [0] Creating conda env tta_patho ==="
    bash scripts/setup_env.sh
  fi
  conda activate tta_patho
fi
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"

MODELS="resnet50 efficientnet_b3 swin_t vit_b16 phikon_v2"
SEEDS="17 31 53 97 211"; CNN_EXTRA="307 401"

# ---- [3] evaluation run: full grid with all fixes (resumable, preserves results) ----
# Note: Existing completed evaluation runs in results/ and run_registry.sqlite
# are NEVER overwritten or deleted. Any run with an existing JSON result file is skipped.
echo "=== [3] evaluation run (resumable, preserving existing results) ==="
python run_evaluation.py || echo "  parent exited non-zero — continuing to ablations"

# ---- [4] ablations (single seed, center 0 -- fast) ----
echo "=== [4] Ablations ==="
for B in 16 32 64 128; do
  python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model resnet50 --method tent  --eval_bs $B --tag bs$B || true
  python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model vit_b16  --method cotta --eval_bs $B --tag bs$B || true
done
python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model resnet50 --method cotta --reset --tag episodic || true
python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model vit_b16  --method tent  --reset --tag episodic || true
for STP in 1 2 4 8; do
  python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model resnet50 --method tent --steps $STP --tag steps$STP || true
  python run_evaluation.py --dataset camelyon17 --domain 0 --seed 17 --model vit_b16  --method sar  --steps $STP --tag steps$STP || true
done

# ---- [5] analysis + figures + corrected tables ----
echo "=== [5] Analysis + figures + tables ==="
python analysis/run_all_analysis.py || true
python postprocess/make_outputs.py || true
python figures/generate_all_figures.py || true

echo "============================================================"
echo "  DONE.  results/  |  figures/output_v2/  |  .../tables/"
echo "============================================================"
python run_evaluation.py --status
