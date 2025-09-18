# db_helpers.py
import sqlite3, time

DB_NAME = "humanhelper.db"

def safe_execute(query, params=(), fetch=False, retries=5, wait_time=3):
    """Execute SQL safely with retry + countdown when DB is locked."""
    attempt = 0
    while attempt < retries:
        try:
            conn = sqlite3.connect(DB_NAME, timeout=10)
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall() if fetch else None
            conn.commit()
            conn.close()
            return rows
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                wait = wait_time * (attempt + 1)
                print(f"⚠️ DB busy, retrying in {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)  # countdown delay
                attempt += 1
            else:
                raise e
    raise Exception("Database still locked after retries ❌")
