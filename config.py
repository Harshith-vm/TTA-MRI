# config.py
"""
Central configuration. ALL paths anchor to the repository root, computed from
THIS file's location — never from the current working directory. This guarantees
that no matter where you launch the pipeline from (and whatever the drive is
mounted as on Linux, e.g. /mnt/usb/MRI), every artifact is written back beside
the code on the same disk.

Also holds the experiment grid and A6000 hardware settings.
"""
import os
import platform
from pathlib import Path
import torch

_IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Paths — all relative to the repo root (this file's parent), NOT os.getcwd()
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

DATA_ROOT      = ROOT / "data" / "raw"
DATASETS_ROOT  = ROOT / "datasets"
CKPT_ROOT      = ROOT / "checkpoints"
RESULTS_ROOT   = ROOT / "results"
FIGURES_ROOT   = ROOT / "figures" / "output"
SALIENCY_ROOT  = ROOT / "results" / "saliency"
REGISTRY_DB    = ROOT / "results" / "run_registry.sqlite"
RESULTS_JSONL  = RESULTS_ROOT / "all_results.jsonl"
AGG_ROOT       = RESULTS_ROOT / "aggregated"

for _p in (DATA_ROOT, CKPT_ROOT, RESULTS_ROOT, FIGURES_ROOT, SALIENCY_ROOT, AGG_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

# Export so legacy modules that read os.environ still resolve to this disk.
os.environ.setdefault("CKPT_ROOT", str(CKPT_ROOT))
os.environ.setdefault("MRI_ROOT", str(ROOT))

# ---------------------------------------------------------------------------
# Hardware — RTX A6000 (48 GB, Ampere). BF16 is numerically safer than FP16.
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16          # Ampere supports bf16 natively
USE_COMPILE = not _IS_WINDOWS       # torch.compile unsupported on Windows (no Triton)
# 16 workers × per-run fork of the 150 MB pandas metadata + timm model was
# the dominant RAM cost across the sweep. 4 workers still saturates the A6000
# with prefetch=2 and keeps the base-process footprint reasonable.
NUM_WORKERS = 4 if _IS_WINDOWS else 4
PIN_MEMORY = True

# Per-model eval batch sizes tuned for 48 GB. Foundation ViTs are large.
BATCH_SIZE = {
    "resnet50":        256,
    "efficientnet_b3": 192,
    "swin_t":          192,
    "vit_b16":         128,
    "phikon_v2":       64,
    "h_optimus_0":     32,
    "prov_gigapath":   32,
}
def batch_size(model_name: str) -> int:
    return BATCH_SIZE.get(model_name, 64)

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
MODELS = [
    "resnet50", "efficientnet_b3",         # CNN baselines
    "swin_t", "vit_b16",                    # Transformer baselines
    "phikon_v2",                            # Pathology foundation model (ViT-L)
    # The two 1.1B models below are disabled for the ~1-week run budget; they are
    # largely redundant with Phikon for the explanation-stability question. Re-add
    # to run the full 7-model grid (adds ~2 weeks).
    # "h_optimus_0", "prov_gigapath",
]

TTA_METHODS = [
    "no_tta", "bn_adapt", "tent", "eata", "sar",
    "cotta", "rotta", "t3a", "deyo",
    "lame", "roid", "sotta", "vida",       # 2022-2024 SOTA additions
    "come", "rem", "sicl", "rmemsafe",     # 2025-2026 SOTA additions
]

XAI_METHODS = [
    # Integrated Gradients is the uniform PRIMARY (architecture-agnostic, valid
    # on every model incl. the foundation ViT-L). Grad-CAM/++ kept as secondary
    # for the 4 non-foundation nets. ScoreCAM dropped from the main run (10-50x
    # slower); re-add for a robustness appendix if desired.
    "integrated_gradients", "gradcam", "gradcampp", "transformer_attribution",
]

# Datasets: 'binary' = Camelyon17 centers, 'multiclass' = NCT-CRC.
DATASETS = {
    "camelyon17": {"kind": "binary",     "num_classes": 2, "domains": [0, 1, 2, 3, 4]},
    "nct_crc":    {"kind": "multiclass", "num_classes": 9, "domains": ["val7k"]},
}

# Non-famous primes — avoids the "you cherry-picked 42" objection.
SEEDS = [17, 31, 53, 97, 211]
# CNNs get 2 extra seeds (head+BN variance exceeds frozen-backbone ViTs).
EXTRA_CNN_SEEDS = [307, 401]
def seeds_for(model_name: str):
    if model_name in ("resnet50", "efficientnet_b3"):
        return SEEDS + EXTRA_CNN_SEEDS
    return SEEDS

N_MC_PASSES = 20                 # MC-dropout uncertainty passes
N_SALIENCY_PER_BATCH = 16        # saliency maps sampled per saliency-batch
N_SALIENCY_BATCHES = 4           # only compute saliency on the first N batches/run
N_BOOTSTRAP = 2000               # bootstrap CI resamples

# Evaluate AUC/ECE/ESI on a fixed random subset of each test domain (keeps tight
# CIs while bounding runtime). Same subset across all models/methods/seeds for a
# given (dataset, domain) -> paired comparisons stay valid. None = use all.
MAX_EVAL_SAMPLES = 15000
