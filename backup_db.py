#!/usr/bin/env python3
# backup_db.py
# Makes a safe sqlite backup using sqlite3.Connection.backup()
import sqlite3, sys, os
from datetime import datetime

DB_PATH = os.environ.get("HH_DB_PATH", "human_helper.db")  # relative to cwd by default
OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else None

if not OUT_PATH:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_PATH = f"human_helper_backup_{ts}.db"

if not os.path.exists(DB_PATH):
    print("ERROR: DB not found:", DB_PATH, file=sys.stderr)
    sys.exit(2)

# Ensure target folder exists
os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)) or ".", exist_ok=True)

# Use sqlite3 backup API (atomic, safe even with WAL)
src = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
dest = sqlite3.connect(OUT_PATH)
with dest:
    src.backup(dest, pages=0, progress=None)
dest.close()
src.close()
print(OUT_PATH)
