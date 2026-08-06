#!/bin/bash
# scripts/apply_to_disk.sh — run on the LINUX lab machine.
# Syncs this code-only staging tree onto the hard disk without touching the
# datasets / checkpoints / results already on the drive.
#
# Usage: bash apply_to_disk.sh /mnt/usb/MRI
set -e
DEST="${1:?Usage: apply_to_disk.sh <path-to-MRI-on-drive>}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing code from $SRC  ->  $DEST"
rsync -av \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='datasets' --exclude='data/raw' \
  --exclude='checkpoints' --exclude='results' --exclude='figures/output' \
  "$SRC"/ "$DEST"/

chmod +x "$DEST"/scripts/*.sh
echo "Done. Next:  cd $DEST && bash scripts/setup_env.sh && bash scripts/run_all.sh"
