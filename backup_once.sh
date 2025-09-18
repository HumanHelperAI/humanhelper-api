#!/usr/bin/env bash
# backup_once.sh
# Usage: ./backup_once.sh /path/to/pendrive/BackupRoot
set -euo pipefail

BASE_TARGET="${1:-}"   # pendrive mount folder root where backups placed
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
DB_PATH="${DB_PATH:-$PROJECT_DIR/human_helper.db}"

if [ -z "$BASE_TARGET" ]; then
  echo "Usage: $0 /path/to/pendrive/BackupRoot"
  exit 2
fi

if [ ! -d "$BASE_TARGET" ]; then
  echo "Target not found: $BASE_TARGET" >&2
  exit 3
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEST_DIR="$BASE_TARGET/humanhelper_backup/$TIMESTAMP"
mkdir -p "$DEST_DIR"

echo "[backup_once] creating safe sqlite backup..."
# call python backup script (writes backup file into dest dir)
python3 backup_db.py "$DEST_DIR/human_helper_${TIMESTAMP}.db"

echo "[backup_once] rsyncing project (excludes .venv and logs)..."
# rsync excludes: venv, node_modules, large temp files, logs (optional)
rsync -a --delete \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='logs' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  "$PROJECT_DIR/" "$DEST_DIR/project/"

# create/update 'latest' symlink
ln -snf "$DEST_DIR" "$BASE_TARGET/humanhelper_backup/latest"

echo "[backup_once] done -> $DEST_DIR"


# prune: keep latest 7 timestamped dirs
KEEP=7
parent="$BASE_TARGET/humanhelper_backup"
if [ -d "$parent" ]; then
  ls -1d "$parent"/*/ 2>/dev/null | sort -r | awk "NR>$KEEP" | xargs -r rm -rf
fi
