#!/bin/sh
set -e

# Default DB location inside container (persist via volume at /data)
: "${DATABASE_URL:=/data/db.sqlite3}"
: "${DB_PATH:=$DATABASE_URL}"

export DATABASE_URL DB_PATH

# Ensure data directory exists and is writable
mkdir -p /data
chmod 0775 /data || true

# Initialize DB (safe: init_db uses CREATE TABLE IF NOT EXISTS)
python3 - <<'PY'
try:
    from database import init_db, DB_NAME
    print("Initializing DB at:", DB_NAME)
    init_db()
except Exception as e:
    # print to stdout/stderr so container logs record it
    import sys
    print("DB init error:", e, file=sys.stderr)
PY

# Run the app with Gunicorn (2 workers). If you prefer dev server, replace with flask run.
exec gunicorn -w 2 -b 0.0.0.0:5000 app:app
