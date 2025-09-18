#!/usr/bin/env bash
set -euo pipefail

# check_pendrive.sh - simple, portable read/write micro-benchmark for Termux
# Usage:
#   ./check_pendrive.sh              # auto-detect a candidate mount (may pick internal)
#   ./check_pendrive.sh /storage/0F43-725E   # explicitly test a mount

HOME_TMP="${HOME}/.hh_tmp"
mkdir -p "$HOME_TMP"

TEST_SIZE_MB=16
BS=4096

# Optional user-provided path (explicit pendrive path)
if [ $# -ge 1 ]; then
  CANDIDATE="$1"
else
  # common candidate locations; we'll pick first directory that exists and is writable
  CANDIDATES=(
    "$HOME/storage/external-1"   # termux external-1
    "/storage/0F43-725E"        # example (replace with your id)
    "/storage/OF43-725E"        # alternate pattern
    "/storage"                  # fallback
    "/sdcard"
    "$HOME/storage"
  )
  CANDIDATE=""
  for c in "${CANDIDATES[@]}"; do
    if [ -d "$c" ] && [ -w "$c" ]; then
      CANDIDATE="$c"
      break
    fi
  done
  # If still empty, pick first writable subdir under ~/storage
  if [ -z "$CANDIDATE" ] && [ -d "$HOME/storage" ]; then
    for d in "$HOME/storage"/*; do
      [ -d "$d" ] || continue
      if [ -w "$d" ]; then CANDIDATE="$d"; break; fi
    done
  fi
fi

if [ -z "${CANDIDATE:-}" ]; then
  echo "No writable candidate mount found. Please pass the mount path explicitly:"
  echo "  ./check_pendrive.sh /storage/0F43-725E"
  exit 1
fi

echo "Testing mount: $CANDIDATE"
echo
echo "Top-level listing (first 40 lines):"
ls -lah "$CANDIDATE" | sed -n '1,40p' || true
echo
echo "Disk usage for this mount:"
df -h "$CANDIDATE" || true
echo

TEST_FILE="$CANDIDATE/hh_test_file.bin"
COUNT=$(( TEST_SIZE_MB * 1024 * 1024 / BS ))

WRITE_LOG="$HOME_TMP/dd_write.log"
READ_LOG="$HOME_TMP/dd_read.log"

echo "Creating ${TEST_SIZE_MB}MB test file: $TEST_FILE"
start_write=$(date +%s.%N)

# prefer oflag=direct when available
if dd --help 2>/dev/null | grep -q oflag; then
  dd if=/dev/zero of="$TEST_FILE" bs=$BS count=$COUNT oflag=direct conv=fsync 2> "$WRITE_LOG" || \
    dd if=/dev/zero of="$TEST_FILE" bs=$BS count=$COUNT conv=fsync 2> "$WRITE_LOG"
else
  dd if=/dev/zero of="$TEST_FILE" bs=$BS count=$COUNT conv=fsync 2> "$WRITE_LOG"
fi

end_write=$(date +%s.%N)
write_time=$(awk "BEGIN {printf \"%.6f\", $end_write - $start_write}")
write_speed=$(awk "BEGIN {if ($write_time > 0) printf \"%.2f\", $TEST_SIZE_MB / $write_time; else print \"inf\"}")

echo "Write time: ${write_time}s — approx ${write_speed} MB/s"
echo "dd write output (last 10 lines):"
tail -n 10 "$WRITE_LOG" || true
echo

echo "Read test (reading file back to /dev/null)..."
start_read=$(date +%s.%N)

# prefer iflag=direct when available
if dd --help 2>/dev/null | grep -q iflag; then
  dd if="$TEST_FILE" of=/dev/null bs=$BS count=$COUNT iflag=direct 2> "$READ_LOG" || \
    dd if="$TEST_FILE" of=/dev/null bs=$BS count=$COUNT 2> "$READ_LOG"
else
  dd if="$TEST_FILE" of=/dev/null bs=$BS count=$COUNT 2> "$READ_LOG"
fi

end_read=$(date +%s.%N)
read_time=$(awk "BEGIN {printf \"%.6f\", $end_read - $start_read}")
read_speed=$(awk "BEGIN {if ($read_time > 0) printf \"%.2f\", $TEST_SIZE_MB / $read_time; else print \"inf\"}")

echo "Read time: ${read_time}s — approx ${read_speed} MB/s"
echo "dd read output (last 10 lines):"
tail -n 10 "$READ_LOG" || true
echo

echo "Removing test file..."
rm -f "$TEST_FILE"
echo "Done. Summary:"
df -h "$CANDIDATE" || true
echo "Write ~ ${write_speed} MB/s, Read ~ ${read_speed} MB/s"
echo "Temporary logs are at: $HOME_TMP (dd_write.log / dd_read.log)"
