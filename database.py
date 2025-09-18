# database.py
import sqlite3
import time
import os

DB_NAME = os.getenv("DATABASE_URL", os.getenv("DB_PATH", "db.sqlite3"))
if DB_NAME.startswith("sqlite:///"):
    DB_NAME = DB_NAME.replace("sqlite:///", "")
DB_NAME = os.path.abspath(DB_NAME)


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # Users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        mobile TEXT UNIQUE,
        aadhar TEXT UNIQUE,
        pan TEXT UNIQUE,
        password TEXT,
        balance REAL DEFAULT 0
    )
    """)

    # Transactions (basic)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mobile TEXT,
        type TEXT,
        amount REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Watch history
    cur.execute("""
    CREATE TABLE IF NOT EXISTS watch_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mobile TEXT,
        video_id TEXT,
        content_type TEXT,
        duration INTEGER,
        reward REAL,
        charity REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Charity wallet (single row)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS charity_wallet (
        id INTEGER PRIMARY KEY CHECK (id=1),
        balance REAL DEFAULT 0
    )
    """)
    cur.execute("INSERT OR IGNORE INTO charity_wallet (id, balance) VALUES (1,0)")

    # Audit log
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT,
        actor TEXT,
        target TEXT,
        details TEXT,
        ip TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Add optional columns (safe)
    try:
        cur.execute("ALTER TABLE users ADD COLUMN is_banned INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    try:
        cur.execute("ALTER TABLE transactions ADD COLUMN status TEXT DEFAULT 'completed'")
    except sqlite3.OperationalError:
        pass

    try:
        cur.execute("ALTER TABLE transactions ADD COLUMN admin_note TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


# ------------------------
# Housekeeping
# ------------------------
def cleanup_old_logs():
    """Delete transactions & watch_history older than 30 days (safe-ish)."""
    try:
        rows, _ = safe_execute("DELETE FROM transactions WHERE timestamp < DATE('now','-30 day')")
        rows, _ = safe_execute("DELETE FROM watch_history WHERE timestamp < DATE('now','-30 day')")
        print("🧹 Old logs cleanup done")
    except Exception as e:
        if "locked" in str(e).lower():
            print("⚠️ Skipping cleanup, DB is locked (will try next time).")
        else:
            raise


# ------------------------
# Connection Helpers
# ------------------------
def get_connection():
    """Open DB with WAL mode for concurrency"""
    conn = sqlite3.connect(DB_NAME, timeout=8, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=8000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def safe_execute(query, params=(), fetch=False, retries=3, base_delay=2):
    """
    Run SQL safely with retries if DB is locked.
    Returns: (rows, countdown_messages)
    """
    countdown_msgs = []
    for attempt in range(retries):
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall() if fetch else None
            conn.commit()
            conn.close()
            return rows, countdown_msgs
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                wait_time = base_delay * (attempt + 1)
                countdown_msgs.append(f"⚠️ DB locked. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                raise
    raise Exception("Database too busy ❌ Please try again later")


def fetch_query(query, params=()):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def execute_query(query, params=(), fetch=False):
    rows, _ = safe_execute(query, params, fetch=fetch)
    return rows


def run_query(query, params=(), fetch=False, retries=5, delay=0.2):
    """Simple retry fallback (read or write)."""
    import sqlite3 as _sq
    for i in range(retries):
        try:
            conn = sqlite3.connect(DB_NAME, timeout=5)
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall() if fetch else None
            conn.commit()
            conn.close()
            return rows
        except _sq.OperationalError as e:
            if "locked" in str(e) and i < retries - 1:
                time.sleep(delay)
            else:
                raise
