#!/usr/bin/env bash
set -e
[ -f .env ] && set -o allexport && source .env && set +o allexport
DB="${DB_PATH:-human_helper.db}"
OUT="backups/backup_$(date +%Y%m%d_%H%M%S).sql"

mkdir -p backups

if [ ! -f "$DB" ]; then
  echo "No DB at $DB — nothing to back up."
  exit 0
fi

sqlite3 "$DB" ".backup '$DB.bak'"
sqlite3 "$DB" .dump > "$OUT"
echo "Backup created: $OUT"
