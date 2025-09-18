#!/usr/bin/env bash
# copy_or_move_bigfiles.sh
# Usage:
#  ./copy_or_move_bigfiles.sh <src_dir> <dst_dir> [threshold] [--mode copy|move]
# Examples:
#  ./copy_or_move_bigfiles.sh ~/models /storage/0F43-725E/BigFiles 200M --mode copy
#  ./copy_or_move_bigfiles.sh ~/models /storage/0F43-725E/BigFiles 200M --mode move

set -euo pipefail
SRC="${1:-}"
DST="${2:-}"
THRESH="${3:-200M}"
MODE="copy"
if [ "${4:-}" = "--mode" ] || [ "${4:-}" = "--mode=move" ] || [ "${4:-}" = "--mode=copy" ]; then
  # support both styles
  if [[ "${4}" == "--mode" ]]; then MODE="${5:-copy}"; else MODE="${4#--mode=}"; fi
fi

if [ -z "$SRC" ] || [ -z "$DST" ]; then
  echo "Usage: $0 <src_dir> <dst_dir> [threshold] [--mode copy|move]"
  exit 2
fi

if [[ "$MODE" != "copy" && "$MODE" != "move" ]]; then
  echo "Invalid mode: $MODE (must be copy or move)"
  exit 2
fi

mkdir -p "$DST"
LOG="$PWD/transfer_$(date +%Y%m%d_%H%M%S).log"
echo "Starting transfer mode=$MODE" | tee -a "$LOG"
echo "Searching for files > $THRESH in: $SRC" | tee -a "$LOG"

# find files >= threshold (human-friendly), use -size for find; -print0 for safe names
files_found=0
while IFS= read -r -d '' file; do
  files_found=$((files_found+1))
  fsize=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")
  fname="$(basename "$file")"
  dstpath="$DST/$fname"

  echo "📂 Processing: $file -> $dstpath" | tee -a "$LOG"

  # Create dst dir if nested
  mkdir -p "$(dirname "$dstpath")"

  # copy with rsync for resume/progress
  echo "   -> copying..." | tee -a "$LOG"
  rsync -a --progress --checksum --partial "$file" "$dstpath" 2>&1 | tee -a "$LOG"

  # compute sha256 on both sides (use sha256sum or shasum)
  echo "   -> verifying checksum..." | tee -a "$LOG"
  if command -v sha256sum >/dev/null 2>&1; then
    srcsum=$(sha256sum "$file" | awk '{print $1}')
    dstsum=$(sha256sum "$dstpath" | awk '{print $1}')
  else
    srcsum=$(shasum -a 256 "$file" | awk '{print $1}')
    dstsum=$(shasum -a 256 "$dstpath" | awk '{print $1}')
  fi

  if [ "$srcsum" != "$dstsum" ]; then
    echo "   ❌ checksum mismatch for $file (src:$srcsum dst:$dstsum)" | tee -a "$LOG"
    echo "   -> removing incomplete dest and continuing" | tee -a "$LOG"
    rm -f "$dstpath"
    continue
  fi

  echo "   ✅ Checksum OK: $srcsum" | tee -a "$LOG"

  # If move mode, remove source after verifying
  if [ "$MODE" = "move" ]; then
    echo "   -> removing source $file" | tee -a "$LOG"
    rm -f "$file"
  fi

done < <(find "$SRC" -type f -size +"$THRESH" -print0)

echo "📊 Summary:" | tee -a "$LOG"
echo "Files processed: $files_found" | tee -a "$LOG"
echo "Logs: $LOG" | tee -a "$LOG"
