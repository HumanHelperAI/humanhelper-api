#!/usr/bin/env bash
# pendrive_watcher.sh
# Watches /storage for newly mounted external storage (polling).
# When a writable mount is detected, it runs backup_once.sh <mount>/HumanHelperBackups
#
# Configure:
#  - POLL_INTERVAL: seconds between checks
#  - MOUNT_BASES: paths to check (default /storage and /mnt/media_rw)
set -euo pipefail

POLL_INTERVAL=${POLL_INTERVAL:-5}
MOUNT_BASES=${MOUNT_BASES:-"/storage /mnt/media_rw"}

# Where inside the mount to store backups (folder will be created)
REMOTE_SUBDIR=${REMOTE_SUBDIR:-"HumanHelperBackups"}

# A simple marker so we don't repeatedly backup same mount (keeps last mount path)
LAST_BACKUP_MARKER="${HOME}/.last_pendrive_backup"

echo "[watcher] starting; polling every ${POLL_INTERVAL}s"

while true; do
  found=""
  for base in $MOUNT_BASES; do
    # list directories in each base (skip emulated primary at /storage/emulated)
    if [ -d "$base" ]; then
      for d in "$base"/*; do
        # candidate likely to be a mount point when it is a directory and writable
        if [ -d "$d" ] && [ -w "$d" ]; then
          # skip emulated primary path common on Android (/storage/emulated or 'self')
          case "$d" in
            */emulated*|*/self) continue ;;
          esac
          # Many external drives appear as /storage/XXXX-XXXX
          # Use first writable candidate
          found="$d"
          break 2
        fi
      done
    fi
  done

  if [ -n "$found" ]; then
    echo "[watcher] detected candidate mount: $found"
    # Compose target root
    TARGET_ROOT="$found/$REMOTE_SUBDIR"
    mkdir -p "$TARGET_ROOT" || true

    # dedupe: read last marker
    last=""
    if [ -f "$LAST_BACKUP_MARKER" ]; then
      last="$(cat "$LAST_BACKUP_MARKER" 2>/dev/null || true)"
    fi

    if [ "$last" != "$found" ]; then
      echo "[watcher] new drive -> running backup_once"
      # run backup (project dir assumed to be current script directory)
      cd "$(dirname "$0")" || true
      ./backup_once.sh "$TARGET_ROOT" || echo "[watcher] backup_once failed"
      echo "$found" > "$LAST_BACKUP_MARKER"
      echo "[watcher] backup finished for $found"
    else
      # optionally still create daily backup if date changed
      echo "[watcher] drive already backed in last run: $found"
    fi
  fi

  sleep "$POLL_INTERVAL"
done
