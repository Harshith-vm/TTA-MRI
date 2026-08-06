<#
.SYNOPSIS
    run_all_windows.ps1 — Full TTA × Explainability pipeline for Windows (A6000 lab).

.DESCRIPTION
    Windows PowerShell replacement for run_all.sh + setup_env.sh + apply_to_disk.sh.
    Run from any directory — it resolves all paths from its own location.

    What it does (in order):
      0. Creates / activates a conda environment  (tta_patho, Python 3.10)
      1. Downloads / verifies datasets & pretrained weights
      2. Source-domain training  (PCam → Camelyon17, NCT-CRC-100K → 7K)
      3. TTA evaluation  (all model × TTA × XAI combos, resumable)
      4. Statistical analysis
      5. Publication figures
      6. Copies results + checkpoints + figures to an external drive

.NOTES
    Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_all_windows.ps1
            (or right-click → "Run with PowerShell")
#>

# ── Strict mode ──────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"

# ── Resolve repo root from this script's location ───────────────────────────
$REPO_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  TTA x Explainability Pipeline  —  Windows A6000 Runner"       -ForegroundColor Cyan
Write-Host "  Repo root: $REPO_ROOT"                                        -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ── A6000 / Ampere performance environment variables ────────────────────────
$env:PYTHONUNBUFFERED       = "1"
$env:TORCH_CUDNN_V8_API_ENABLED = "1"
$env:NVIDIA_TF32_OVERRIDE  = "1"
$env:OMP_NUM_THREADS        = "8"

# ── Helper: run a command, abort on failure ──────────────────────────────────
function Invoke-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][scriptblock]$Action
    )
    Write-Host ("`n>> " + $Label) -ForegroundColor Yellow
    & $Action
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        $msg = "  FAILED with exit code " + $LASTEXITCODE + ". Aborting."
        Write-Host $msg -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ── Helper: find conda ──────────────────────────────────────────────────────
function Get-CondaActivateCommand {
    # Try common conda install paths
    $condaPaths = @(
        "$env:USERPROFILE\miniconda3",
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\Miniconda3",
        "$env:USERPROFILE\Anaconda3",
        "C:\ProgramData\miniconda3",
        "C:\ProgramData\anaconda3",
        "C:\tools\miniconda3",
        "C:\tools\Miniconda3"
    )
    foreach ($p in $condaPaths) {
        $hook = Join-Path $p "shell\condabin\conda-hook.ps1"
        if (Test-Path $hook) { return $hook }
    }
    # Fallback: try conda on PATH
    $condaExe = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaExe) {
        $condaRoot = Split-Path (Split-Path $condaExe.Source)
        $hook = Join-Path $condaRoot "shell\condabin\conda-hook.ps1"
        if (Test-Path $hook) { return $hook }
    }
    return $null
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — Conda environment
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "[0/6] Setting up conda environment..." -ForegroundColor Green

$condaHook = Get-CondaActivateCommand
if (-not $condaHook) {
    Write-Host "ERROR: Could not locate conda. Install Miniconda/Anaconda first." -ForegroundColor Red
    exit 1
}

# Initialize conda for this PowerShell session
& $condaHook

# Create env if it doesn't exist
$envExists = conda env list 2>&1 | Select-String -Pattern "tta_patho"
if (-not $envExists) {
    Write-Host "  Creating conda env 'tta_patho' (Python 3.10)..." -ForegroundColor DarkCyan
    conda create -n tta_patho python=3.10 -y
}

conda activate tta_patho

# Install PyTorch + CUDA 12.1 (Ampere/A6000, bf16 + torch.compile)
Write-Host "  Installing PyTorch 2.2 + CUDA 12.1..." -ForegroundColor DarkCyan
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 `
    -c pytorch -c nvidia -y

# Core scientific stack
Write-Host "  Installing scientific stack..." -ForegroundColor DarkCyan
conda install numpy=1.26.4 scipy=1.11.4 scikit-learn=1.3.2 pandas=2.1.4 `
    matplotlib=3.8.2 seaborn=0.13.0 pillow=10.1.0 h5py=3.10.0 tqdm=4.66.1 `
    -c conda-forge -y

# Pip packages
Write-Host "  Installing pip packages..." -ForegroundColor DarkCyan
pip install `
    timm==0.9.16 `
    transformers==4.40.0 `
    grad-cam==1.5.0 `
    captum==0.7.0 `
    einops==0.7.0 `
    scikit-image==0.22.0 `
    statsmodels==0.14.1 `
    pingouin==0.5.4 `
    scikit-posthocs==0.9.0 `
    huggingface-hub==0.23.0 `
    accelerate==0.29.0 `
    albumentations==1.4.0 `
    opencv-python-headless==4.9.0.80 `
    torchmetrics==1.3.0 `
    wilds==2.0.0

# Sanity check
Write-Host "  GPU sanity check..." -ForegroundColor DarkCyan
python -c "import torch; n=torch.cuda.get_device_name(0); print('GPU:', n); assert torch.cuda.is_available(); print('bf16:', torch.cuda.is_bf16_supported())"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: GPU not available or bf16 not supported. Check CUDA install." -ForegroundColor Red
    exit 1
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Download / verify datasets
# ══════════════════════════════════════════════════════════════════════════════
Invoke-Step -Label "[1/6] Downloading / verifying datasets..." -Action {
    Set-Location $REPO_ROOT
    python data/download_all.py
}
# Verify is non-fatal (some model weights download on first use)
Write-Host "  Verifying pretrained weights..." -ForegroundColor DarkCyan
Set-Location $REPO_ROOT
python scripts/verify_downloads.py 2>&1 | Write-Host

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Source-domain training
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[2/6] Source-domain training (PCam->Camelyon17, NCT-CRC-100K->7K)..." -ForegroundColor Green

$MODELS = @("resnet50", "efficientnet_b3", "swin_t", "vit_b16",
            "phikon_v2", "h_optimus_0", "prov_gigapath")
$SEEDS  = @(17, 31, 53, 97, 211)
$CNN_EXTRA_SEEDS = @(307, 401)
$DATASETS = @("camelyon17", "nct_crc")

Set-Location $REPO_ROOT
foreach ($ds in $DATASETS) {
    foreach ($model in $MODELS) {
        $seedList = $SEEDS
        if ($model -eq "resnet50" -or $model -eq "efficientnet_b3") {
            $seedList = $SEEDS + $CNN_EXTRA_SEEDS
        }
        foreach ($seed in $seedList) {
            Write-Host "  train: $ds | $model | seed=$seed" -ForegroundColor DarkGray
            python training/train_source.py --model $model --seed $seed --dataset $ds
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  WARNING: training $ds|$model|seed=$seed failed" -ForegroundColor Red
            }
        }
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — TTA evaluation  (resumable, multi-XAI, faithfulness)
# ══════════════════════════════════════════════════════════════════════════════
Invoke-Step -Label "[3/6] TTA evaluation (resumable; multi-XAI, faithfulness)..." -Action {
    Set-Location $REPO_ROOT
    python run_evaluation.py
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Statistical analysis
# ══════════════════════════════════════════════════════════════════════════════
Invoke-Step -Label "[4/6] Statistical analysis (LMM, paired Wilcoxon+FDR, Friedman/Nemenyi)..." -Action {
    Set-Location $REPO_ROOT
    python analysis/run_all_analysis.py
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Publication figures
# ══════════════════════════════════════════════════════════════════════════════
Invoke-Step -Label "[5/6] Publication figures..." -Action {
    Set-Location $REPO_ROOT
    python figures/generate_all_figures.py
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Copy results to external hard disk
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[6/6] Copying results to external hard disk..." -ForegroundColor Green

# Show final status
Set-Location $REPO_ROOT
python run_evaluation.py --status

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "  Pipeline complete!" -ForegroundColor Cyan
Write-Host "  Results  : $REPO_ROOT\results"
Write-Host "  Figures  : $REPO_ROOT\figures\output"
Write-Host "  Checkpoints: $REPO_ROOT\checkpoints"
Write-Host "================================================================" -ForegroundColor Cyan

# Ask for external drive path
Write-Host ""
$extDrive = Read-Host "Enter path to external hard disk (e.g. E:\MRI_Results) or press Enter to skip"
if ($extDrive -and $extDrive.Trim() -ne "") {
    $extDrive = $extDrive.Trim()

    # Create destination directories
    $destResults = Join-Path $extDrive "results"
    $destFigures = Join-Path $extDrive "figures_output"
    $destCheckpoints = Join-Path $extDrive "checkpoints"
    $destSaliency = Join-Path $extDrive "saliency"

    New-Item -ItemType Directory -Path $destResults -Force | Out-Null
    New-Item -ItemType Directory -Path $destFigures -Force | Out-Null
    New-Item -ItemType Directory -Path $destCheckpoints -Force | Out-Null
    New-Item -ItemType Directory -Path $destSaliency -Force | Out-Null

    Write-Host "`nCopying results..." -ForegroundColor Yellow
    robocopy "$REPO_ROOT\results"           $destResults /E /NP /NFL /NDL /R:2 /W:1
    Write-Host "Copying figures..." -ForegroundColor Yellow
    robocopy "$REPO_ROOT\figures\output"    $destFigures /E /NP /NFL /NDL /R:2 /W:1
    Write-Host "Copying checkpoints..." -ForegroundColor Yellow
    robocopy "$REPO_ROOT\checkpoints"       $destCheckpoints /E /NP /NFL /NDL /R:2 /W:1
    Write-Host "Copying saliency maps..." -ForegroundColor Yellow
    robocopy "$REPO_ROOT\results\saliency"  $destSaliency /E /NP /NFL /NDL /R:2 /W:1

    # robocopy returns 0-7 for "success" codes
    Write-Host "`n================================================================" -ForegroundColor Green
    Write-Host "  All results copied to: $extDrive" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
} else {
    Write-Host "`nSkipped external copy. Results remain in $REPO_ROOT" -ForegroundColor DarkYellow
}

Write-Host "`nDone. Total elapsed time: pipeline finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan