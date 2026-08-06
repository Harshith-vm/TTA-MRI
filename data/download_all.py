# data/download_all.py
"""
Cross-platform dataset downloader (Windows + Linux). Pure Python — no wget /
gunzip / unzip shell calls. Idempotent and resumable: existing files are
skipped, partial downloads are re-fetched.

Datasets:
  - PCam        (Zenodo HDF5, gz)         -> source for the binary Camelyon17 track
  - Camelyon17  (WILDS, ~10.6 GB)         -> shifted target (5 hospital centres)
  - NCT-CRC-HE  (Zenodo zip)              -> 9-class colorectal track
"""
import os
import sys
import gzip
import zipfile
import shutil
import ssl
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

DATA_ROOT = config.DATA_ROOT
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Some lab networks have strict/old TLS; fall back to an unverified context only
# for these large public research files if the default handshake fails.
_DEFAULT_CTX = ssl.create_default_context()
_UNVERIFIED_CTX = ssl._create_unverified_context()


def _download(url, dest_path, chunk=1 << 20):
    """Stream a URL to dest_path with a progress bar and TLS fallback."""
    dest_path = Path(dest_path)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    for ctx in (_DEFAULT_CTX, _UNVERIFIED_CTX):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b)
                        done += len(b)
                        if total:
                            pct = 100 * done / total
                            print(f"\r  {dest_path.name}: {pct:5.1f}%  "
                                  f"({done/1e9:.2f}/{total/1e9:.2f} GB)", end="", flush=True)
            print()
            tmp.replace(dest_path)
            return True
        except Exception as e:
            print(f"\n  [{url.split('/')[-1]}] failed via "
                  f"{'verified' if ctx is _DEFAULT_CTX else 'unverified'} TLS: {e}")
            if tmp.exists():
                tmp.unlink()
    return False


def _gunzip(gz_path, out_path):
    with gzip.open(gz_path, "rb") as fi, open(out_path, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    gz_path.unlink()


def download_pcam():
    dest = DATA_ROOT / "pcam"
    dest.mkdir(parents=True, exist_ok=True)
    base = "https://zenodo.org/record/2546921/files/"
    names = [
        "camelyonpatch_level_2_split_train_x.h5",
        "camelyonpatch_level_2_split_train_y.h5",
        "camelyonpatch_level_2_split_valid_x.h5",
        "camelyonpatch_level_2_split_valid_y.h5",
        "camelyonpatch_level_2_split_test_x.h5",
        "camelyonpatch_level_2_split_test_y.h5",
    ]
    for n in names:
        out = dest / n
        if out.exists():
            print(f"PCam {n} present, skip.")
            continue
        gz = dest / (n + ".gz")
        print(f"Downloading PCam {n} ...")
        if not _download(base + n + ".gz", gz):
            raise RuntimeError(f"Could not download {n}. See manual instructions.")
        print(f"  decompressing {n} ...")
        _gunzip(gz, out)
    print("PCam ready.")


def download_camelyon17():
    dest = DATA_ROOT / "wilds"
    # WILDS ships camelyon17_v1.0; presence of metadata.csv = complete.
    if (dest / "camelyon17_v1.0" / "metadata.csv").exists():
        print("Camelyon17-WILDS already present, skip.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import wilds
        wilds.get_dataset("camelyon17", root_dir=str(dest), download=True)
        print("Camelyon17-WILDS ready.")
    except Exception as e:
        print(f"\n*** Camelyon17 auto-download failed: {e}")
        print("*** This is almost always a NETWORK/TLS issue on the 10.6 GB file.")
        print("*** Manual fix: download the archive from")
        print("***   https://worksheets.codalab.org/bundles/0xe45e15f39fb54e9d9e919556af67aabe")
        print(f"***   save as {dest/'camelyon17_v1.0'/'archive.tar.gz'} and extract it there,")
        print("***   then re-run this script (it will skip everything already done).")


def download_nct_crc():
    dest = config.DATASETS_ROOT / "nct_crc"
    if (dest / "NCT-CRC-HE-100K").exists() and (dest / "CRC-VAL-HE-7K").exists():
        print("NCT-CRC already present, skip.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in [
        ("NCT-CRC-HE-100K.zip", "https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip"),
        ("CRC-VAL-HE-7K.zip",   "https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip"),
    ]:
        zpath = dest / name
        if not (dest / name.replace(".zip", "")).exists():
            print(f"Downloading {name} ...")
            if not _download(url, zpath):
                raise RuntimeError(f"Could not download {name}.")
            print(f"  extracting {name} ...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(dest)
            zpath.unlink()
    print("NCT-CRC ready.")


if __name__ == "__main__":
    download_nct_crc()      # already on your disk -> skips instantly
    download_pcam()         # needed for the binary Camelyon17 track
    download_camelyon17()   # large; network-dependent
    print("\nAll datasets processed.")
