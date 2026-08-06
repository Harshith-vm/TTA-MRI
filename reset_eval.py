# reset_eval.py — clear the OLD (buggy) evaluation outputs so the fp32 re-run
# regenerates them. KEEPS training checkpoints and source-training registry rows,
# so no model is retrained. Old results are MOVED to a backup (never deleted).
import sqlite3, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

res = config.RESULTS_ROOT
backup = res.parent / "results_bf16_backup"
backup.mkdir(parents=True, exist_ok=True)

for name in ["all_results.jsonl", "raw", "saliency", "aggregated"]:
    src = res / name
    if not src.exists():
        continue
    dst = backup / name
    try:
        if dst.exists():
            (shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink(missing_ok=True))
        shutil.move(str(src), str(dst))
        print(f"  moved {name} -> {dst}")
    except Exception as e:
        print(f"  warning backing up {name}: {e}")

(res / "raw").mkdir(parents=True, exist_ok=True)
(res / "aggregated").mkdir(parents=True, exist_ok=True)
config.SALIENCY_ROOT.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(config.REGISTRY_DB) as c:
    n = c.execute("DELETE FROM runs WHERE stage LIKE 'tta_evaluation%'").rowcount
    c.commit()
print(f"cleared {n} eval registry rows; checkpoints + training rows kept.")
print(f"old (bf16) results backed up at: {backup}")
