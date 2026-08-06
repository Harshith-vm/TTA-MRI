# -*- coding: utf-8 -*-
"""
download_datasets.py
====================
Downloads all datasets used in the TTA Explainability Pipeline for Histopathology.

Datasets:
  1. PatchCamelyon (PCam)        — Zenodo HDF5 files (~8 GB total)
  2. Camelyon17-WILDS            — via wilds Python package (~455K patches)
  3. NCT-CRC-HE-100K + CRC-VAL-HE-7K — Zenodo ZIP files (~4.5 GB)

Windows-safe: uses requests + gzip + zipfile (no wget/gunzip/unzip needed).
Resume-safe:  skips files that already exist.
Progress:     shows download progress with tqdm.

Usage:
    python datasets/download_datasets.py [--skip-pcam] [--skip-wilds] [--skip-nct]
"""

import os
import sys
import gzip
import shutil
import zipfile
import argparse
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check and install
# ---------------------------------------------------------------------------
def ensure_pkg(pkg, import_name=None):
    import importlib
    import subprocess
    import_name = import_name or pkg
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[setup] Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

ensure_pkg("requests")
ensure_pkg("tqdm")

import io
import requests

# Windows: reconfigure stdout to UTF-8 so progress bars and symbols render correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASETS_ROOT = Path(__file__).parent  # d:\MRI\datasets\
PCAM_DIR     = DATASETS_ROOT / "pcam"
WILDS_DIR    = DATASETS_ROOT / "wilds"
NCT_DIR      = DATASETS_ROOT / "nct_crc"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_file(url: str, dest_path: Path, desc: str = None,
                  max_retries: int = 10) -> Path:
    """
    Download a file with:
      - HTTP Range-request resumption  (keeps .part file between runs/crashes)
      - Automatic retry with exponential backoff on connection drops
      - tqdm progress bar showing speed and ETA
    Skips entirely if dest_path already exists.
    """
    import time

    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"  [skip] already exists: {dest_path.name}")
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    label = desc or dest_path.name
    print(f"  [download] {label}")
    print(f"             from: {url}")
    print(f"             to:   {dest_path}")

    for attempt in range(1, max_retries + 1):
        # Detect how many bytes we already have (range-request resume)
        resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0

        headers = {}
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"
            print(f"  [resume] attempt {attempt}/{max_retries} — resuming from "
                  f"{resume_pos / 1e6:.1f} MB")
        elif attempt > 1:
            print(f"  [retry]  attempt {attempt}/{max_retries}")

        try:
            with requests.get(url, stream=True, headers=headers,
                              timeout=60) as r:
                if r.status_code == 416:
                    # Server says range not satisfiable — file already complete
                    tmp_path.rename(dest_path)
                    print(f"  [ok] {dest_path.name} already fully downloaded.")
                    return dest_path

                r.raise_for_status()
                # Total size from Content-Range or Content-Length
                total_size = int(r.headers.get("content-length", 0)) + resume_pos

                mode = "ab" if resume_pos > 0 else "wb"
                with open(tmp_path, mode) as f, tqdm(
                    total=total_size if total_size > 0 else None,
                    initial=resume_pos,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=label, leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}]"
                ) as bar:
                    for chunk in r.iter_content(chunk_size=512 * 1024):  # 512 KB
                        f.write(chunk)
                        bar.update(len(chunk))

            # Success — rename to final path
            tmp_path.rename(dest_path)
            size_gb = dest_path.stat().st_size / 1e9
            print(f"  [ok] saved {dest_path.name} ({size_gb:.2f} GB)")
            return dest_path

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                OSError) as e:
            wait = min(2 ** attempt, 120)  # exponential backoff, max 2 min
            if attempt < max_retries:
                print(f"\n  [warn] Connection error (attempt {attempt}/{max_retries}): "
                      f"{type(e).__name__}")
                print(f"         .part file kept — will resume. Waiting {wait}s ...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed after {max_retries} attempts: {url}\n"
                    f"  Partial file kept at: {tmp_path}\n"
                    f"  Re-run the script to resume."
                ) from e



def gunzip_file(gz_path: Path, out_path: Path):
    """Decompress a .gz file. Skips if output already exists."""
    if out_path.exists():
        print(f"  [skip] already decompressed: {out_path.name}")
        return
    print(f"  [decompress] {gz_path.name} → {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1 << 23)  # 8 MB buffer
    gz_path.unlink()  # free disk space after decompressing
    print(f"  [ok] decompressed {out_path.name}")


def unzip_file(zip_path: Path, dest_dir: Path):
    """Extract a ZIP archive. Skips if dest_dir already has content."""
    print(f"  [extract] {zip_path.name} → {dest_dir}/")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.infolist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}", leave=False):
            z.extract(member, dest_dir)
    zip_path.unlink()  # free disk space
    print(f"  [ok] extracted to {dest_dir}")


# ---------------------------------------------------------------------------
# Dataset 1: PatchCamelyon (PCam) — Zenodo HDF5
# ---------------------------------------------------------------------------
PCAM_FILES = [
    "camelyonpatch_level_2_split_train_x.h5.gz",
    "camelyonpatch_level_2_split_train_y.h5.gz",
    "camelyonpatch_level_2_split_valid_x.h5.gz",
    "camelyonpatch_level_2_split_valid_y.h5.gz",
    "camelyonpatch_level_2_split_test_x.h5.gz",
    "camelyonpatch_level_2_split_test_y.h5.gz",
]
PCAM_ZENODO = "https://zenodo.org/record/2546921/files/"


def download_pcam():
    print("\n" + "=" * 60)
    print("Dataset 1: PatchCamelyon (PCam)")
    print("  License: CC0 | Size: ~8 GB | 6 HDF5 files")
    print("=" * 60)

    # Check sentinel: if all 6 .h5 files are present, skip entirely
    h5_files = [PCAM_DIR / f.replace(".gz", "") for f in PCAM_FILES]
    if all(f.exists() for f in h5_files):
        print("  [skip] All PCam HDF5 files already present.\n")
        return

    PCAM_DIR.mkdir(parents=True, exist_ok=True)

    for fname in PCAM_FILES:
        h5_name = fname.replace(".gz", "")
        h5_path = PCAM_DIR / h5_name

        if h5_path.exists():
            print(f"  [skip] {h5_name} already decompressed.")
            continue

        gz_path = PCAM_DIR / fname
        url = PCAM_ZENODO + fname
        download_file(url, gz_path, desc=fname)
        gunzip_file(gz_path, h5_path)

    print("\n  PCam download complete!")
    print(f"  Location: {PCAM_DIR}")
    for f in sorted(PCAM_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name} ({size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# Dataset 2: Camelyon17-WILDS
# ---------------------------------------------------------------------------
def download_camelyon17_wilds():
    print("\n" + "=" * 60)
    print("Dataset 2: Camelyon17-WILDS")
    print("  License: CC0 | Size: ~10 GB | 5-hospital patches")
    print("=" * 60)

    sentinel = WILDS_DIR / "camelyon17_v2.0"
    if sentinel.exists():
        print("  [skip] Camelyon17-WILDS already downloaded.\n")
        return

    try:
        import wilds
    except ImportError:
        import subprocess
        print("  [setup] Installing wilds package ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wilds", "-q"])
        import wilds

    WILDS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading via wilds package to {WILDS_DIR} ...")
    print("  (This may take a while — ~10 GB download)")

    dataset = wilds.get_dataset("camelyon17", root_dir=str(WILDS_DIR), download=True)
    print(f"\n  Camelyon17-WILDS download complete!")
    print(f"  Location: {WILDS_DIR}")
    print(f"  Total samples: {len(dataset)}")


# ---------------------------------------------------------------------------
# Dataset 3: NCT-CRC-HE-100K + CRC-VAL-HE-7K
# ---------------------------------------------------------------------------
NCT_ZENODO_BASE = "https://zenodo.org/record/1214456/files/"
NCT_FILES = {
    "NCT-CRC-HE-100K.zip": "NCT-CRC-HE-100K",  # zip_name: expected_subdir
    "CRC-VAL-HE-7K.zip":   "CRC-VAL-HE-7K",
}


def download_nct_crc():
    print("\n" + "=" * 60)
    print("Dataset 3: NCT-CRC-HE-100K + CRC-VAL-HE-7K")
    print("  License: CC0 | Size: ~4.5 GB | 9-class CRC tissue patches")
    print("=" * 60)

    # Check sentinel: both subdirs present with .tif files
    all_present = all(
        (NCT_DIR / subdir).exists() and
        len(list((NCT_DIR / subdir).glob("**/*.tif"))) > 100
        for subdir in NCT_FILES.values()
    )
    if all_present:
        print("  [skip] NCT-CRC datasets already downloaded.\n")
        return

    NCT_DIR.mkdir(parents=True, exist_ok=True)

    for zip_name, subdir in NCT_FILES.items():
        dest_subdir = NCT_DIR / subdir
        if dest_subdir.exists() and len(list(dest_subdir.glob("**/*.tif"))) > 100:
            print(f"  [skip] {subdir} already extracted.")
            continue

        zip_path = NCT_DIR / zip_name
        url = NCT_ZENODO_BASE + zip_name
        download_file(url, zip_path, desc=zip_name)
        unzip_file(zip_path, NCT_DIR)

    print("\n  NCT-CRC download complete!")
    print(f"  Location: {NCT_DIR}")
    for d in sorted(NCT_DIR.iterdir()):
        if d.is_dir():
            n_tifs = len(list(d.glob("**/*.tif")))
            print(f"    {d.name}/ — {n_tifs} .tif patches")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary():
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    datasets = {
        "PCam (HDF5)": PCAM_DIR,
        "Camelyon17-WILDS": WILDS_DIR,
        "NCT-CRC-HE-100K + CRC-VAL-HE-7K": NCT_DIR,
    }

    total_gb = 0.0
    for name, path in datasets.items():
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            gb = size / 1e9
            total_gb += gb
            status = f"[OK] {gb:.2f} GB"
        else:
            status = "[--] not downloaded"
        print(f"  {name:<40} {status}")

    print(f"\n  Total on disk: {total_gb:.2f} GB")
    print(f"  Root: {DATASETS_ROOT}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all histopathology datasets")
    parser.add_argument("--skip-pcam",  action="store_true", help="Skip PatchCamelyon download")
    parser.add_argument("--skip-wilds", action="store_true", help="Skip Camelyon17-WILDS download")
    parser.add_argument("--skip-nct",   action="store_true", help="Skip NCT-CRC download")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TTA Histopathology Pipeline - Dataset Downloader")
    print("  All datasets: CC0 / public domain, no registration")
    print("=" * 60)
    print(f"\nDatasets will be saved to: {DATASETS_ROOT}\n")

    if not args.skip_pcam:
        download_pcam()
    else:
        print("\n[--skip-pcam] Skipping PatchCamelyon.")

    if not args.skip_wilds:
        download_camelyon17_wilds()
    else:
        print("\n[--skip-wilds] Skipping Camelyon17-WILDS.")

    if not args.skip_nct:
        download_nct_crc()
    else:
        print("\n[--skip-nct] Skipping NCT-CRC.")

    print_summary()
    print("All requested datasets are ready!\n")
