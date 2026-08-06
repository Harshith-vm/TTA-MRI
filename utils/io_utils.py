# utils/io_utils.py
import json
import os
import time
import tempfile
from pathlib import Path


def _retry_io(fn, retries=3, delay=1.0):
    """Retry a write closure on transient EIO from the USB enclosure —
    9 vida runs previously died with `OSError: [Errno 5]` on write."""
    last = None
    for i in range(retries):
        try:
            return fn()
        except OSError as e:
            last = e
            time.sleep(delay * (i + 1))
    raise last


def atomic_json_save(data, path):
    """Write JSON atomically — no corruption on crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    def _do():
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic + overwrites; works on Windows and Linux
    _retry_io(_do)


def atomic_torch_save(state_dict, path):
    """Save PyTorch checkpoint atomically."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    def _do():
        torch.save(state_dict, tmp)
        os.replace(tmp, path)  # atomic + overwrites; works on Windows and Linux
    _retry_io(_do)


def append_jsonl(data, path):
    """Append a result record to a JSONL file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def _do():
        with open(path, 'a') as f:
            f.write(json.dumps(data, default=str) + '\n')
    _retry_io(_do)

def load_json_safe(path, default=None):
    """Load JSON — return default if file missing or corrupt."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default
