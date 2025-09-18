#!/usr/bin/env bash
set -e
[ -f .env ] && set -o allexport && source .env && set +o allexport
DB="${DB_PATH:-human_helper.db}"

LAST_DUMP=$(ls -1 backups/backup_*.sql 2>/dev/null | tail -n 1 || true)
if [ -z "$LAST_DUMP" ]; then
  echo "No backup dumps found in ./backups"
  exit 1
fi

echo "Restoring from: $LAST_DUMP"
rm -f "$DB"
sqlite3 "$DB" < "$LAST_DUMP"
echo "Restore complete ✅"
