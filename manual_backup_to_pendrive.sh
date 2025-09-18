#!/usr/bin/env bash
# manual_backup_to_pendrive.sh /path/to/mount
if [ -z "${1:-}" ]; then
  echo "Usage: $0 /path/to/mount"
  exit 2
fi
TARGET="$1/HumanHelperBackups"
mkdir -p "$TARGET"
cd "$(dirname "$0")"
./backup_once.sh "$TARGET"
