
#!/usr/bin/env python3
import os
import sqlite3

DB = os.environ.get("DATABASE_URL") or "db.sqlite3"
# If DATABASE_URL is of form "sqlite:///path", extract path
if DB.startswith("sqlite:///"):
    DB = DB.replace("sqlite:///", "")

print("Using DB file:", DB)
conn = sqlite3.connect(DB)
cur = conn.cursor()

sql = """
PRAGMA foreign_keys=OFF;

CREATE TABLE IF NOT EXISTS transactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mobile TEXT,
  type TEXT,
  amount REAL,
  status TEXT,
  admin_note TEXT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rewards (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mobile TEXT,
  source TEXT,
  amount REAL,
  charity_amount REAL DEFAULT 0,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mobile TEXT,
  type TEXT,
  payload TEXT,
  status TEXT DEFAULT 'new',
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rides (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mobile TEXT,
  pickup TEXT,
  dropoff TEXT,
  status TEXT DEFAULT 'requested',
  fare_estimate REAL DEFAULT 0,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

VACUUM;
"""

cur.executescript(sql)
conn.commit()
conn.close()
print("Tables created/ensured.")

# create_tables.py
from database import init_db
if __name__ == "__main__":
    init_db()
    print("DB tables created / ensured.")
