#!/usr/bin/env python3
# app.py - consolidated HumanHelper backend (all-in-one, defensive)

# stdlib
import os
import time
import json
import random
import threading
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Tuple, Optional, Any, Dict, List

# third-party
import requests
from flask import Flask, request, jsonify, Blueprint, g
from flask_cors import CORS
import jwt
from passlib.hash import pbkdf2_sha256
from werkzeug.exceptions import HTTPException

# DB helpers (sqlite always available; psycopg2 imported only when using Postgres)
import sqlite3
from urllib.parse import urlparse



# Basic logger so import-time messages are visible
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("humanhelper")



# -----------------------
# DATABASE_URL handling
# -----------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db.sqlite3") or "sqlite:///db.sqlite3"
DATABASE_URL = DATABASE_URL.strip()

# Normalize the old `postgres://` scheme (some libraries expect `postgresql://`)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

# Detect engine
IS_PG = DATABASE_URL.startswith(("postgresql://", "postgres://"))

# Optional: tidy sqlite path form if you ever use it directly
if DATABASE_URL.startswith("sqlite:///"):
    SQLITE_PATH = DATABASE_URL[len("sqlite:///"):]
else:
    SQLITE_PATH = None

# Try to import psycopg2 only when we intend to use PG
if IS_PG:
    try:
        import psycopg2
        import psycopg2.extras
        logger.info("[db] psycopg2 available; Postgres support enabled ✅")
    except Exception as exc:
        # Keep IS_PG=False so runtime falls back to sqlite in this environment,
        # but log a clear message so you know why Postgres won't be used.
        logger.warning("[db] psycopg2 import failed — Postgres disabled. Install psycopg2-binary in requirements.txt. Error: %s", exc)
        IS_PG = False


def get_db():
    """
    Return a connection object with a consistent interface on both engines:
    - conn.cursor() -> cursor
    - cursor.execute(sql, params)
    - cursor.fetchone() / .fetchall() -> dict-like rows
    - conn.commit(), conn.rollback(), conn.close()
    """
    if IS_PG:
        # Railway Postgres (require SSL)
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        conn.autocommit = False

        # Wrap to always return RealDictCursor cursors (dict rows)
        class _PgConn:
            def __init__(self, c):
                self._c = c

            def cursor(self):
                return self._c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            def commit(self):
                self._c.commit()

            def rollback(self):
                self._c.rollback()

            def close(self):
                self._c.close()

            # convenience parity with sqlite3.Connection.execute(...)
            def execute(self, sql, params=()):
                cur = self.cursor()
                cur.execute(sql, params)
                return cur

        return _PgConn(conn)

    # SQLite (dev / Termux) fallback
    db_path = DATABASE_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # dict-like rows
    return conn


# ---- Local dev safety: stub any missing helpers ----
def _noop(*a, **k):
    pass


for _name in [
    "ensure_live_schema",
    "ensure_user_columns",
    "ensure_auth_tables",
    "ensure_schema_migrations",
    "ensure_wallet_tables",
    "cleanup_old_logs",
]:
    if _name not in globals():
        globals()[_name] = _noop


# ---------- Schema ensure (idempotent) ----------
def ensure_live_schema_pg():
    """Create/upgrade tables in Postgres. Safe to run repeatedly."""
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
          id               SERIAL PRIMARY KEY,
          name             TEXT,
          mobile           TEXT UNIQUE,
          email            TEXT,
          address          TEXT,
          password_hash    TEXT,
          verification_code TEXT,
          verify_expires   BIGINT,
          is_verified      INTEGER DEFAULT 0,
          is_banned        INTEGER DEFAULT 0,
          balance          NUMERIC(12,2) DEFAULT 0,
          locked_balance   NUMERIC(12,2) DEFAULT 0,
          created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens(
          jti        TEXT PRIMARY KEY,
          user_id    INTEGER NOT NULL REFERENCES users(id),
          issued_at  BIGINT NOT NULL,
          expires_at BIGINT NOT NULL,
          revoked    INTEGER NOT NULL DEFAULT 0
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_exp  ON refresh_tokens(expires_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_rev  ON refresh_tokens(revoked)")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS wallet_txns (
          id             SERIAL PRIMARY KEY,
          user_id        INTEGER NOT NULL REFERENCES users(id),
          kind           TEXT NOT NULL,
          amount         NUMERIC(12,2) NOT NULL,
          balance_after  NUMERIC(12,2) NOT NULL,
          locked_after   NUMERIC(12,2) NOT NULL,
          status         TEXT NOT NULL,
          ref            TEXT,
          note           TEXT,
          meta           JSONB,
          created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_user    ON wallet_txns(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_created ON wallet_txns(created_at)")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS withdrawal_requests(
          id         SERIAL PRIMARY KEY,
          user_id    INTEGER NOT NULL REFERENCES users(id),
          amount     NUMERIC(12,2) NOT NULL,
          net_amount NUMERIC(12,2),
          fee_amount NUMERIC(12,2),
          upi        TEXT,
          payout_id  TEXT,
          status     TEXT NOT NULL DEFAULT 'requested',
          reason     TEXT,
          method     TEXT,
          account    TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wreq_user   ON withdrawal_requests(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wreq_status ON withdrawal_requests(status)")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS fee_pool (
          id            INTEGER PRIMARY KEY,
          pool_balance  NUMERIC(12,2) NOT NULL DEFAULT 0,
          updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("INSERT INTO fee_pool (id, pool_balance) VALUES (1,0) ON CONFLICT (id) DO NOTHING")

        db.commit()
        print("[migrate:pg] ✅ schema ensured")
    except Exception as e:
        db.rollback()
        print("[migrate:pg] ⚠️", e)
        raise
    finally:
        db.close()


def ensure_live_schema_sqlite():
    """Create/upgrade tables in SQLite. Safe to run repeatedly."""
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT,
          mobile TEXT UNIQUE,
          email TEXT,
          address TEXT,
          password_hash TEXT,
          verification_code TEXT,
          verify_expires INTEGER,
          is_verified INTEGER DEFAULT 0,
          is_banned  INTEGER DEFAULT 0,
          balance REAL DEFAULT 0,
          locked_balance REAL DEFAULT 0,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens(
          jti TEXT PRIMARY KEY,
          user_id INTEGER NOT NULL,
          issued_at INTEGER NOT NULL,
          expires_at INTEGER NOT NULL,
          revoked INTEGER NOT NULL DEFAULT 0
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_exp  ON refresh_tokens(expires_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_rev  ON refresh_tokens(revoked)")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS wallet_txns (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          kind TEXT NOT NULL,
          amount REAL NOT NULL,
          balance_after REAL NOT NULL,
          locked_after REAL NOT NULL,
          status TEXT NOT NULL,
          ref TEXT,
          note TEXT,
          meta TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_user    ON wallet_txns(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_txn_created ON wallet_txns(created_at)")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS withdrawal_requests(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          amount REAL NOT NULL,
          net_amount REAL,
          fee_amount REAL,
          upi TEXT,
          payout_id TEXT,
          status TEXT NOT NULL DEFAULT 'requested',
          reason TEXT,
          method TEXT,
          account TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wreq_user   ON withdrawal_requests(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wreq_status ON withdrawal_requests(status)")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS fee_pool (
          id INTEGER PRIMARY KEY CHECK (id=1),
          pool_balance REAL NOT NULL DEFAULT 0,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("INSERT OR IGNORE INTO fee_pool (id, pool_balance) VALUES (1, 0)")

        db.commit()
        print("[migrate:sqlite] ✅ schema ensured")
    except Exception as e:
        db.rollback()
        print("[migrate:sqlite] ⚠️", e)
        raise
    finally:
        db.close()


# Run the right migrator on startup (once)
try:
    if IS_PG:
        ensure_live_schema_pg()
    else:
        ensure_live_schema_sqlite()
except Exception as e:
    print("[migrate] init warning:", e)


print(f"[db] engine={'postgres' if IS_PG else 'sqlite'} url={DATABASE_URL.split('@')[-1][:48]}…")


# ---- Cross-DB SQL helpers ----
def _fix_placeholders(sql: str) -> str:
    """
    Convert SQLite-style '?' params to Postgres '%s' placeholders.
    Only replaces ? characters that are outside of single/double quotes.
    """
    if not IS_PG:
        return sql

    out = []
    in_single = False
    in_double = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
        elif ch == "?" and not in_single and not in_double:
            out.append("%s")
        else:
            out.append(ch)
        i += 1
    return "".join(out)



def insert_ret_id(db, sql: str, params: tuple = ()) -> int:
    """
    Run an insert and return the inserted id in a DB-agnostic way.
    For Postgres we append RETURNING id when missing.
    For SQLite we return lastrowid or 0.
    """
    sql2 = _fix_placeholders(sql)
    if IS_PG:
        if "returning" not in sql2.lower():
            sql2 = sql2.rstrip("; ") + " RETURNING id"
        cur = db.execute(sql2, params)
        row = cur.fetchone()
        # psycopg RealDictCursor returns a dict-like row
        return int(row["id"] if isinstance(row, dict) else row[0])
    else:
        cur = db.execute(sql2, params)
        # sqlite3 cursor: use lastrowid on the connection's cursor object
        try:
            return int(getattr(cur, "lastrowid", 0) or 0)
        except Exception:
            return 0










# ----------------------                                                              
# Flask app init
# ----------------------
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded

app = Flask(__name__)
app.url_map.strict_slashes = False

# --- Production-safe defaults (paste near app init) ---
import os, secrets
app.config.setdefault("ENV", os.getenv("FLASK_ENV", "production"))
app.config.setdefault("DEBUG", False)
app.config.setdefault("TESTING", False)
app.config.setdefault("SECRET_KEY", os.getenv("SECRET_KEY", os.getenv("JWT_SECRET", secrets.token_urlsafe(64))))
app.config.setdefault("SESSION_COOKIE_SECURE", True)
app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")


from flask_talisman import Talisman

TALISMAN_CONFIG = {
    "force_https": True,
    "strict_transport_security": True,
    "strict_transport_security_max_age": 31536000,  # 1 year
    # minimal CSP; tighten as needed for your static assets
    "content_security_policy": {
        "default-src": ["'self'"],
        "script-src": ["'self'"],
        "style-src": ["'self'"],
        "img-src": ["'self' data:"],
    },
    "frame_options": "DENY"
}
Talisman(app, **TALISMAN_CONFIG)

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

redis_url = os.getenv("RATE_LIMIT_REDIS", "redis://127.0.0.1:6379/0")
limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url, default_limits=["200 per day", "50 per hour"])
limiter.init_app(app)


# Use REDIS_URL if present, else fallback to in-memory (dev)
RATE_LIMIT_STORAGE_URI = os.getenv("REDIS_URL") or os.getenv("RATE_LIMIT_REDIS_URI")

def rate_key():
    """
    Per-user limits when authenticated, otherwise per-IP.
    """
    if getattr(g, "user_id", None):
        return f"user:{g.user_id}"
    return get_remote_address()

limiter = Limiter(
    key_func=rate_key,
    app=app,
    default_limits=["1000/day", "200/hour"],
    storage_uri=RATE_LIMIT_STORAGE_URI,   # e.g. redis://default:<password>@<host>:<port>/0
    strategy="fixed-window",
    headers_enabled=True
)

@app.errorhandler(RateLimitExceeded)
def ratelimit_handler(e):
    return jsonify({
        "ok": False,
        "error": {
            "code": "rate_limited",
            "message": "Too many requests. Try later."
        }
    }), 429









# ----------------------
# Fallback DB helpers (if `database` module not present)
# ----------------------
try:
    from database import init_db, run_query, cleanup_old_logs
except Exception:
    print("[hh] Warning: database module not found — using SQLite helpers")

    def init_db():
        db = get_db()
        cur = db.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            mobile TEXT UNIQUE,
            password_hash TEXT,
            email TEXT,
            address TEXT,
            is_banned INTEGER DEFAULT 0,
            balance REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        db.commit()
        db.close()

    def run_query(sql: str, params: tuple = (), fetch: bool = False):
        db = get_db()
        cur = db.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() if fetch else None
        db.commit()
        db.close()
        return rows










# =========================
# AUTO MIGRATION + SELF-HEAL
# =========================
import sqlite3

def _table_exists(cur, table: str) -> bool:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,))
    return cur.fetchone() is not None

def _col_exists(cur, table: str, col: str) -> bool:
    try:
        cur.execute(f"PRAGMA table_info({table})")
        return any(r[1] == col for r in cur.fetchall())
    except Exception:
        return False

# ---------- AUTO MIGRATION (Postgres) ----------
def ensure_live_schema_pg():
    """Create/upgrade tables for Postgres. Safe to run repeatedly."""
    db = get_db()
    cur = db.cursor()
    try:
        # users
        execq(db, """
        CREATE TABLE IF NOT EXISTS users (
            id              SERIAL PRIMARY KEY,
            name            TEXT,
            mobile          TEXT UNIQUE,
            password_hash   TEXT,
            email           TEXT,
            address         TEXT,
            is_verified     INTEGER DEFAULT 0,
            is_banned       INTEGER DEFAULT 0,
            balance         NUMERIC(12,2) DEFAULT 0,
            locked_balance  NUMERIC(12,2) DEFAULT 0,
            verification_code TEXT,
            verify_expires  BIGINT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        execq(db, "ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT")
        execq(db, "ALTER TABLE users ADD COLUMN IF NOT EXISTS address TEXT")
        execq(db, "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified INTEGER DEFAULT 0")
        execq(db, "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_banned INTEGER DEFAULT 0")
        execq(db, "ALTER TABLE users ADD COLUMN IF NOT EXISTS balance NUMERIC(12,2) DEFAULT 0")
        execq(db, "ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_balance NUMERIC(12,2) DEFAULT 0")
        execq(db, "ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_code TEXT")
        execq(db, "ALTER TABLE users ADD COLUMN IF NOT EXISTS verify_expires BIGINT")

        # refresh_tokens
        execq(db, """
        CREATE TABLE IF NOT EXISTS refresh_tokens(
            jti        TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            issued_at  BIGINT NOT NULL,
            expires_at BIGINT NOT NULL,
            revoked    INTEGER NOT NULL DEFAULT 0
        )
        """)
        execq(db, "CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id)")
        execq(db, "CREATE INDEX IF NOT EXISTS idx_refresh_exp  ON refresh_tokens(expires_at)")
        execq(db, "CREATE INDEX IF NOT EXISTS idx_refresh_rev  ON refresh_tokens(revoked)")

        # wallet_txns
        execq(db, """
        CREATE TABLE IF NOT EXISTS wallet_txns (
            id             SERIAL PRIMARY KEY,
            user_id        INTEGER NOT NULL REFERENCES users(id),
            kind           TEXT NOT NULL,
            amount         NUMERIC(12,2) NOT NULL,
            balance_after  NUMERIC(12,2) NOT NULL,
            locked_after   NUMERIC(12,2) NOT NULL,
            status         TEXT NOT NULL,
            ref            TEXT,
            note           TEXT,
            meta           JSONB,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        execq(db, "CREATE INDEX IF NOT EXISTS idx_txn_user    ON wallet_txns(user_id)")
        execq(db, "CREATE INDEX IF NOT EXISTS idx_txn_created ON wallet_txns(created_at)")

        # withdrawal_requests
        execq(db, """
        CREATE TABLE IF NOT EXISTS withdrawal_requests(
            id         SERIAL PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            amount     NUMERIC(12,2) NOT NULL,
            net_amount NUMERIC(12,2),
            fee_amount NUMERIC(12,2),
            upi        TEXT,
            payout_id  TEXT,
            status     TEXT NOT NULL DEFAULT 'requested',
            reason     TEXT,
            method     TEXT,
            account    TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
        """)
        execq(db, "CREATE INDEX IF NOT EXISTS idx_wreq_user   ON withdrawal_requests(user_id)")
        execq(db, "CREATE INDEX IF NOT EXISTS idx_wreq_status ON withdrawal_requests(status)")

        # fee_pool (single bucket)
        execq(db, """
        CREATE TABLE IF NOT EXISTS fee_pool (
            id           INTEGER PRIMARY KEY,
            pool_balance NUMERIC(12,2) NOT NULL DEFAULT 0,
            updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        execq(db, "INSERT INTO fee_pool (id, pool_balance) VALUES (1,0) ON CONFLICT (id) DO NOTHING")

        db.commit()
        print("[migrate:pg] ✅ schema ensured")
    except Exception as e:
        db.rollback()
        print("[migrate:pg] ⚠️", e)
        raise
    finally:
        db.close()

# Call exactly once at boot:
try:
    if IS_PG:
        ensure_live_schema_pg()
    else:
        ensure_live_schema()  # your SQLite function from the snippet you posted
except Exception as e:
    print("[migrate] init warning:", e)


# Heal on the next request if a schema error pops up
def _repair_fee_pool():
    db = get_db()
    cur = db.cursor()
    try:
        if IS_PG:
            execq(db, """
            CREATE TABLE IF NOT EXISTS fee_pool (
                id           INTEGER PRIMARY KEY,
                pool_balance NUMERIC(12,2) NOT NULL DEFAULT 0,
                updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            execq(db, "INSERT INTO fee_pool (id, pool_balance) VALUES (1,0) ON CONFLICT (id) DO NOTHING")
        else:
            # SQLite path
            cur.execute("""
            CREATE TABLE IF NOT EXISTS fee_pool (
                id INTEGER PRIMARY KEY CHECK (id=1),
                pool_balance REAL NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            cur.execute("INSERT OR IGNORE INTO fee_pool (id, pool_balance) VALUES (1,0)")
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


AUTO_SCHEMA_PATTERNS = (
    "no such table",
    "no such column",
    "has no column",
    "unknown column",
    'relation "',          # PG: relation "table" does not exist
    "does not exist",
    "column "
)

def _maybe_fix_schema_from_error(e: Exception):
    msg = (str(e) or "").lower()
    try:
        if any(p in msg for p in AUTO_SCHEMA_PATTERNS):
            if IS_PG:
                ensure_live_schema_pg()
            else:
                ensure_live_schema()
    except Exception:
        # swallow in auto-fix; original handler will still respond
        pass

# When SQLite schema errors bubble up, auto-fix for next request
if not IS_PG:
    @app.errorhandler(sqlite3.OperationalError)
    def _sqlite_op_error(e):
        _maybe_fix_schema_from_error(e)
        # Return a clean JSON; next request will usually succeed
        return jsonify({"ok": False, "error": {"code": "internal_error", "message": "Internal server error"}}), 500

# Also run occasionally to keep things tidy
@app.before_request
def _bg_schema_tick():
    # tiny chance to re-check on live traffic
    if random.randint(1, 200) == 1:
        try:
            if IS_PG:
                ensure_live_schema_pg()
            else:
                ensure_live_schema()
        except Exception:
            pass











# ------------------------
# Small helper: unify param style between sqlite and psycopg2
# ------------------------
# ---------- DB exec helper (works for both sqlite & psycopg2) ----------
def db_exec(db_conn, sql: str, params: tuple | list = ()):
    """
    Engine-agnostic SQL executor.
    - Use %s placeholders in SQL. When running on SQLite this function will
      translate %s -> ? automatically. On Postgres it leaves %s as-is.
    - db_conn may be:
        * sqlite3.Connection
        * sqlite3.Cursor
        * our _PgConn wrapper (connection-like with .cursor())
        * psycopg2 cursor
    Returns a cursor-like object (so caller can .fetchone()/.fetchall()).
    """
    if params is None:
        params = ()

    # If db_conn exposes .cursor(), assume it's a connection-like object.
    # Otherwise assume it's already a cursor and use it directly.
    cur = None
    conn_object = None
    try:
        # prefer to detect connection-like objects that provide cursor()
        if hasattr(db_conn, "cursor"):
            conn_object = db_conn
            cur = db_conn.cursor()
        else:
            # caller passed a cursor already
            cur = db_conn
    except Exception:
        # fallback: treat db_conn as cursor
        cur = db_conn

    # Normalize SQL placeholder style for SQLite vs Postgres
    sql_to_run = sql if IS_PG else sql.replace("%s", "?")

    # Execute. Note: sqlite3.Cursor.execute accepts params as sequence/tuple.
    # psycopg2 cursor.execute accepts tuple as well.
    try:
        cur.execute(sql_to_run, tuple(params))
    except Exception:
        # If the connection object supports .execute directly (some wrappers),
        # try that as a fallback.
        if conn_object is not None and hasattr(conn_object, "execute"):
            try:
                conn_object.execute(sql_to_run, tuple(params))
                # Return a cursor if possible
                try:
                    return conn_object.cursor()
                except Exception:
                    return conn_object
            except Exception:
                pass
        # re-raise original error for logging upstream
        raise

    return cur


def execq(db, sql: str, params: tuple = ()):
    """Execute a query on the given db connection with placeholder fixing."""
    return db.execute(_fix_placeholders(sql), params)










# =========================
# ADMIN BLUEPRINT + ROUTES
# =========================

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# ---------- Admin auth decorator (use ADMIN_TOKEN from env only) ----------
from functools import wraps
from flask import request, jsonify

# Read admin token from env at import time; helper to reload if you update .env
def _get_admin_token():
    # Prefer ADMIN_TOKEN, fall back to empty string so check fails rather than using a default.
    return os.environ.get("ADMIN_TOKEN", "") or ""

# call this if you update .env during runtime and want app to pick it up (optional)
def reload_admin_token_from_env():
    global ADMIN_TOKEN
    ADMIN_TOKEN = _get_admin_token()
    return ADMIN_TOKEN

# initialize ADMIN_TOKEN global (safe: blank if not provided)
ADMIN_TOKEN = _get_admin_token()

def admin_required(f):
    """
    Decorator for admin-only endpoints.
    Requires header X-Admin-Token: <token> which must exactly match ADMIN_TOKEN env var.
    If ADMIN_TOKEN is empty/missing, admin endpoints are disabled (fail-safe).
    """
    @wraps(f)
    def _wrap(*args, **kwargs):
        # Fail safe: if no ADMIN_TOKEN configured, return 403 (don't allow anonymous fallback)
        if not ADMIN_TOKEN:
            return jsonify({"ok": False, "error": {"code": "admin_disabled", "message": "admin endpoints disabled (no ADMIN_TOKEN configured)"}}), 403

        # Accept token either in X-Admin-Token header or admin_token query param (useful for CLI only)
        token = (request.headers.get("X-Admin-Token") or request.args.get("admin_token") or "").strip()
        if not token:
            return jsonify({"ok": False, "error": {"code": "unauthorized", "message": "missing admin token"}}), 401

        # compare constant-time
        try:
            import hmac
            if not hmac.compare_digest(token, ADMIN_TOKEN):
                return jsonify({"ok": False, "error": {"code": "unauthorized", "message": "invalid admin token"}}), 401
        except Exception:
            # fallback to plain compare (should not happen)
            if token != ADMIN_TOKEN:
                return jsonify({"ok": False, "error": {"code": "unauthorized", "message": "invalid admin token"}}), 401

        # authorized
        return f(*args, **kwargs)
    return _wrap



@admin_bp.get("/wallet/fees")
@admin_required
def admin_fees_read():
    db = get_db()
    try:
        row = db.execute("SELECT pool_balance, updated_at FROM fee_pool WHERE id=1").fetchone()
    except Exception as e:
        # Try to fix schema automatically if pool_balance is missing or table doesn't exist
        db.close()
        _ = admin_fix_fee_pool()
        db = get_db()
        row = db.execute("SELECT pool_balance, updated_at FROM fee_pool WHERE id=1").fetchone()
    db.close()
    if not row:
        return jsonify({"ok": False, "error": {"code": "not_found", "message": "fee pool not initialized"}}), 404
    return jsonify({"ok": True, "pool_balance": float(row["pool_balance"]), "updated_at": row["updated_at"]})


@admin_bp.post("/wallet/fees/transfer")
@admin_required
def admin_fees_transfer():
    """
    Decrease the shared fee pool by `amount` (for charity/ops payouts, etc.).
    Works on both SQLite and Postgres via db_exec() placeholder shim.
    """
    d = request.get_json(silent=True) or {}
    try:
        amt = float(d.get("amount") or 0)
    except Exception:
        amt = 0.0
    note = (d.get("note") or "").strip()

    if amt <= 0:
        return jsonify({"ok": False, "error": {"code": "bad_request", "message": "amount must be > 0"}}), 400

    db = get_db(); cur = db.cursor()
    try:
        # start tx
        db.execute("BEGIN")
        row = db_exec(cur, "SELECT pool_balance FROM fee_pool WHERE id=1").fetchone()
        bal = float(row["pool_balance"] if row else 0.0)

        if bal < amt:
            db.rollback(); db.close()
            return jsonify({
                "ok": False,
                "error": {"code": "insufficient", "message": "insufficient pool balance"}
            }), 400

        db_exec(cur, """
            UPDATE fee_pool
            SET pool_balance = pool_balance - ?,
                updated_at   = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (amt,))

        db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        db.close()
        _maybe_fix_schema_from_error(e)
        return jsonify({"ok": False, "error": {"code": "internal_error", "message": str(e)}}), 500

    db.close()
    return jsonify({"ok": True, "transferred": amt, "note": (note or None)})


# ----- List users (search) -----
@admin_bp.get("/users")
@admin_required
def admin_users_list():
    q = (request.args.get("q") or "").strip()
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    offset = max(int(request.args.get("offset", 0)), 0)

    params, sql = [], "SELECT id,name,mobile,email,balance,is_banned,is_verified,created_at FROM users"
    if q:
        sql += " WHERE name LIKE ? OR mobile LIKE ? OR email LIKE ?"
        like = f"%{q}%"; params.extend([like, like, like])
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = run_query(sql, tuple(params), fetch=True) or []
    users = []
    for r in rows:
        # sqlite Row or tuple support
        id_, name, mobile, email, bal, banned, verified, created = r
        users.append({
            "id": id_,
            "name": name,
            "mobile": mobile,
            "email": email,
            "balance": float(bal or 0),
            "is_banned": bool(banned),
            "is_verified": bool(verified),
            "created_at": created,
        })
    return jsonify({"ok": True, "users": users})


# ----- Ban / Unban user -----
@admin_bp.post("/user/ban")
@admin_required
def admin_user_ban():
    d = request.get_json(silent=True) or {}
    mobile = (d.get("mobile") or "").strip()
    if not mobile:
        return jsonify({"ok": False, "error": {"code": "bad_request", "message": "mobile required"}}), 400
    db = get_db(); cur = db.cursor()
    db_exec(cur, "UPDATE users SET is_banned=1 WHERE mobile=?", (mobile,))
    db.commit(); db.close()
    return jsonify({"ok": True, "message": f"user {mobile} banned"})


@admin_bp.post("/user/unban")
@admin_required
def admin_user_unban():
    d = request.get_json(silent=True) or {}
    mobile = (d.get("mobile") or "").strip()
    if not mobile:
        return jsonify({"ok": False, "error": {"code": "bad_request", "message": "mobile required"}}), 400
    db = get_db(); cur = db.cursor()
    db_exec(cur, "UPDATE users SET is_banned=0 WHERE mobile=?", (mobile,))
    db.commit(); db.close()
    return jsonify({"ok": True, "message": f"user {mobile} unbanned"})


# ----- User transactions (inspect) -----
@admin_bp.get("/user/<int:uid>/txns")
@admin_required
def admin_user_txns(uid: int):
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    offset = max(int(request.args.get("offset", 0)), 0)
    db = get_db()
    rows = db.execute("""
        SELECT id,kind,amount,status,ref,note,meta,created_at,balance_after,locked_after
        FROM wallet_txns WHERE user_id=? ORDER BY id DESC LIMIT ? OFFSET ?
    """, (uid, limit, offset)).fetchall()
    db.close()
    txns = []
    for r in rows:
        txns.append({
            "id": r["id"],
            "kind": r["kind"],
            "amount": float(r["amount"]),
            "status": r["status"],
            "ref": r["ref"],
            "note": r["note"],
            "meta": json.loads(r["meta"]) if r["meta"] else None,
            "balance_after": float(r["balance_after"]),
            "locked_after": float(r["locked_after"]),
            "created_at": r["created_at"],
        })
    return jsonify({"ok": True, "transactions": txns})


# ----- Withdrawals list (monitor only; payouts are automatic) -----
@admin_bp.get("/withdrawals")
@admin_required
def admin_withdrawals_list():
    status = (request.args.get("status") or "").strip().lower()
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    offset = max(int(request.args.get("offset", 0)), 0)

    params = []
    sql = """
      SELECT id, user_id, amount, net_amount, fee_amount, upi, payout_id, status, reason, created_at, updated_at
      FROM withdrawal_requests
    """
    if status:
        sql += " WHERE status=?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    db = get_db()
    rows = db.execute(sql, tuple(params)).fetchall()
    db.close()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "amount": float(r["amount"]),
            "net_amount": float(r["net_amount"]),
            "fee_amount": float(r["fee_amount"]),
            "upi": r["upi"],
            "payout_id": r["payout_id"],
            "status": r["status"],
            "reason": r["reason"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    return jsonify({"ok": True, "withdrawals": out})


# ----- (Optional) credit/debit adjust (for testing only) -----
@admin_bp.post("/user/adjust")
@admin_required
def admin_adjust_balance():
    d = request.get_json()
    user_id = d.get("user_id")
    delta = float(d.get("delta", 0))
    note = d.get("note", "admin adjustment")

    db = get_db()
    cur = db.cursor()
    db_exec(cur, "UPDATE users SET balance = balance + ? WHERE id=?", (delta, user_id))
    db.commit()
    db.close()

    return jsonify({"ok": True, "message": f"Adjusted ₹{delta} for user {user_id}", "note": note})


@admin_bp.post("/db/fix/fee-pool")
@admin_required
def admin_fix_fee_pool():
    """
    One-time repair:
    - Add pool_balance if missing
    - Migrate old charity/maintenance balances into pool_balance
    - Seed row id=1 if missing
    Safe to call repeatedly (idempotent).
    """
    db = get_db(); cur = db.cursor()
    try:
        db.execute("BEGIN IMMEDIATE")

        # ensure table exists
        db_exec(cur, """
            CREATE TABLE IF NOT EXISTS fee_pool (
              id INTEGER PRIMARY KEY CHECK (id=1),
              pool_balance REAL NOT NULL DEFAULT 0,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # detect columns
        cur.execute("PRAGMA table_info(fee_pool)")
        cols = {r[1] for r in cur.fetchall()}

        # add pool_balance if missing
        if "pool_balance" not in cols:
            cur.execute("ALTER TABLE fee_pool ADD COLUMN pool_balance REAL DEFAULT 0")
            cur.execute("PRAGMA table_info(fee_pool)")
            cols = {r[1] for r in cur.fetchall()}

        # seed row
        cur.execute("INSERT OR IGNORE INTO fee_pool (id, pool_balance) VALUES (1, 0)")

        # if old columns exist, migrate sum -> pool_balance
        has_char = "charity_balance" in cols
        has_maint = "maintenance_balance" in cols
        if has_char or has_maint:
            try:
                row = cur.execute(
                    "SELECT "
                    "COALESCE((SELECT charity_balance  FROM fee_pool WHERE id=1),0) AS cb, "
                    "COALESCE((SELECT maintenance_balance FROM fee_pool WHERE id=1),0) AS mb, "
                    "COALESCE((SELECT pool_balance FROM fee_pool WHERE id=1),0) AS pb"
                ).fetchone()
                cb = float(row["cb"]) if row and "cb" in row.keys() else 0.0
                mb = float(row["mb"]) if row and "mb" in row.keys() else 0.0
                pb = float(row["pb"]) if row and "pb" in row.keys() else 0.0
            except Exception:
                cb = mb = 0.0
                row2 = cur.execute("SELECT pool_balance FROM fee_pool WHERE id=1").fetchone()
                pb = float(row2["pool_balance"]) if row2 else 0.0

            total = round(pb if (pb > 0) else (cb + mb), 2)
            db_exec(cur, "UPDATE fee_pool SET pool_balance=?, updated_at=CURRENT_TIMESTAMP WHERE id=1", (total,))

        db.commit()
        db.close()
        return jsonify({"ok": True, "message": "fee_pool repaired"}), 200
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        db.close()
        return jsonify({"ok": False, "error": {"code":"internal_error","message": str(e)}}), 500

@admin_bp.post("/tools/repair-withdrawals")
@admin_required
def admin_repair_withdrawals():
    n = _repair_stuck_withdrawals(max_age_min=0)  # repair immediately
    return jsonify({"ok": True, "released": n})


# --- Admin debug OTP endpoint (GET) ---
# Place this inside the admin blueprint area (near other admin_bp.route handlers)
@admin_bp.route("/debug/otp", methods=["GET"])
@admin_required
def admin_debug_otp():
    """
    Debug-only: return the verification_code and expiry for a mobile.
    Example: GET /admin/debug/otp?mobile=9876509001  (requires X-Admin-Token header)
    """
    mobile = (request.args.get("mobile") or "").strip()
    if not mobile:
        return jsonify({"ok": False, "error": {"code": "bad_request", "message": "mobile query param required"}}), 400

    conn = get_db()
    try:
        # Select correct placeholder for current DB engine
        placeholder = "%s" if IS_PG else "?"
        sql = f"SELECT id, verification_code, verify_expires, is_verified FROM users WHERE mobile = {placeholder}"

        # execq expects the proper placeholder style in SQL
        cur = execq(conn, sql, (mobile,))
        row = cur.fetchone()
        if not row:
            return jsonify({"ok": False, "error": {"code": "not_found", "message": "user not found"}}), 404

        # Row may be dict-like (psycopg2 RealDictCursor) or sqlite3.Row or tuple/list.
        try:
            # Preferred access for dict-like / mapping
            uid = row["id"]
            code = row["verification_code"]
            expires = row["verify_expires"]
            is_verified = bool(row["is_verified"])
        except Exception:
            # Fallback for sequence/tuple-like rows: (id, verification_code, verify_expires, is_verified)
            uid = row[0] if len(row) > 0 else None
            code = row[1] if len(row) > 1 else None
            expires = row[2] if len(row) > 2 else None
            is_verified = bool(row[3]) if len(row) > 3 else False

        return jsonify({
            "ok": True,
            "mobile": mobile,
            "user_id": uid,
            "verification_code": code,
            "expires": expires,
            "is_verified": is_verified
        }), 200

    except Exception as e:
        app.logger.exception("admin_debug_otp: db error")
        return jsonify({"ok": False, "error": {"code": "internal_error", "message": str(e)}}), 500

    finally:
        try:
            conn.close()
        except Exception:
            pass


# ✅ Register admin blueprint once (AFTER all routes above are defined)
app.register_blueprint(admin_bp)












# ----------------------
# Config / env
# ----------------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://api.openrouter.ai/v1/chat/completions")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_REST_URL = os.getenv("GEMINI_REST_URL", "")
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:5001/completion")

# Wallet policy / limits
WALLET_MIN_DEPOSIT = float(os.getenv("WALLET_MIN_DEPOSIT", "10"))
WALLET_MAX_DEPOSIT = float(os.getenv("WALLET_MAX_DEPOSIT", "50000"))
WALLET_MIN_WITHDRAW = float(os.getenv("WALLET_MIN_WITHDRAW", "100"))
WALLET_MAX_WITHDRAW = float(os.getenv("WALLET_MAX_WITHDRAW", "100000"))
WALLET_DAILY_DEPOSIT_CAP = float(os.getenv("WALLET_DAILY_DEPOSIT_CAP", "200000"))
WALLET_DAILY_WITHDRAW_CAP = float(os.getenv("WALLET_DAILY_WITHDRAW_CAP", "200000"))

def _amount_ok(x: Any) -> bool:
    try:
        v = float(x)
        return v == v and abs(v) != float("inf")
    except Exception:
        return False

# timeouts
DEFAULT_REQ_TIMEOUT = int(os.getenv("DEFAULT_REQ_TIMEOUT", "12"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", DEFAULT_REQ_TIMEOUT))
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", DEFAULT_REQ_TIMEOUT))
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", DEFAULT_REQ_TIMEOUT))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", DEFAULT_REQ_TIMEOUT))
LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", DEFAULT_REQ_TIMEOUT))

# free models priority: comma separated like "openrouter:gpt-oss-20b,deepseek:deepseek-v3.1-free"
FREE_MODELS_PRIORITY = [m.strip() for m in os.getenv("FREE_MODELS_PRIORITY", "").split(",") if m.strip()]

# ==== JWT config & helpers ====================================================
if not FREE_MODELS_PRIORITY:
    FREE_MODELS_PRIORITY = [
        "openrouter:gpt-oss-20b",
        "openrouter:gpt-oss-120b",
        "deepseek:deepseek-v3.1-free",
        "deepseek:r1-free",
        "nousresearch:deephermes-3-llama-3-8b-preview",
        "google:gemini-2.0-flash-exp"
    ]

def _now() -> int:
    return int(time.time())

def _make_jti() -> str:
    return uuid.uuid4().hex

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-in-railway")
JWT_ALGO   = "HS256"
ACCESS_TTL_MIN   = int(os.getenv("ACCESS_TTL_MIN", "30"))
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TTL_DAYS", "30"))

def _utcnow():
    return datetime.now(timezone.utc)

def _now_ts():
    return int(_utcnow().timestamp())


def make_access_token(user_id: int, mobile: str) -> str:
    now = _utcnow()
    exp = now + timedelta(minutes=ACCESS_TTL_MIN)
    payload = {
        "type": "access",
        "sub": str(user_id),
        "mobile": mobile,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "jti": str(uuid.uuid4())
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def make_refresh_token(user_id: int) -> str:
    now = _utcnow()
    exp = now + timedelta(days=REFRESH_TTL_DAYS)
    jti = str(uuid.uuid4())
    payload = {
        "type": "refresh",
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "jti": jti
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

    # persist refresh token record using db_exec to handle placeholders
    conn = get_db()
    cur = conn.cursor()
    db_exec(cur, "INSERT INTO refresh_tokens (jti,user_id,issued_at,expires_at,revoked) VALUES (?,?,?,?,0)",
            (jti, user_id, int(now.timestamp()), int(exp.timestamp())))
    conn.commit()
    conn.close()
    return token


def decode_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def auth_required(fn):
    @wraps(fn)
    def _wrap(*a, **k):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "missing token"}), 401
        payload = decode_token(auth.split(" ", 1)[1])
        if not payload or payload.get("type") != "access":
            return jsonify({"error": "invalid or expired token"}), 401
        g.user_id = int(payload["sub"])
        g.mobile = payload.get("mobile") or ""
        return fn(*a, **k)
    return _wrap


def clean_expired_refresh():
    conn = get_db()
    cur = conn.cursor()
    db_exec(cur, "DELETE FROM refresh_tokens WHERE expires_at < ?", (_now(),))
    conn.commit()
    conn.close()
# ==== /JWT ====================================================================





# ----------------------
# Utilities: HTTP with retries/backoff
# ----------------------
def _do_request_with_retries(method: str, url: str, headers: Optional[dict] = None,
                             json_body: Optional[dict] = None,
                             timeout: int = 10, max_attempts: int = 3) -> Tuple[bool, Any]:
    attempt = 0
    backoff_base = 0.5
    last_exc = None
    while attempt < max_attempts:
        try:
            if method.lower() == "post":
                r = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            else:
                r = requests.get(url, headers=headers, params=json_body, timeout=timeout)

            # success
            if 200 <= r.status_code < 300:
                return True, r

            # handle common transient statuses
            if r.status_code in (429, 503):
                # respect Retry-After if present
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        time.sleep(float(ra) + random.uniform(0.1, 0.6))
                    except Exception:
                        time.sleep(backoff_base * (2 ** attempt) + random.uniform(0, 0.3))
                else:
                    time.sleep(backoff_base * (2 ** attempt) + random.uniform(0, 0.3))
                attempt += 1
                continue

            # non-transient HTTP error
            return False, r

        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(backoff_base * (2 ** attempt) + random.uniform(0, 0.3))
            attempt += 1

    return False, f"request failed after {max_attempts} attempts: {last_exc}"


# ----------------------
# Provider wrappers
# ----------------------
def enforce_length(prompt: str, target_words: int | None) -> str:
    if not target_words:
        return prompt
    return (
        f"{prompt}\n\n"
        f"Write **about {target_words} words**. "
        f"Plain paragraph, no lists, no markdown, no code blocks."
    )


def _extract_text_like(j: dict) -> str:
    # 1) OpenAI-like
    try:
        ch = j.get("choices", [])
        if ch:
            first = ch[0]
            # message.content
            msg = (first.get("message") or {})
            txt = msg.get("content")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
            # some providers use "text" directly
            txt = first.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
            # some use "content" as list of parts -> [{"type":"text","text":"..."}]
            content = first.get("content")
            if isinstance(content, list) and content:
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                        return part["text"].strip()
    except Exception:
        pass

    # 2) top-level fallbacks
    for k in ("text", "output", "content", "response"):
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def ask_openrouter(model: str, prompt: str, timeout: int = OPENROUTER_TIMEOUT) -> Tuple[bool, str]:
    if not OPENROUTER_API_KEY or not OPENROUTER_URL:
        return False, "openrouter not configured"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 384,
        "temperature": 0.3
    }
    ok, resp = _do_request_with_retries("post", OPENROUTER_URL, headers=headers, json_body=payload, timeout=timeout)
    if not ok:
        if isinstance(resp, requests.Response):
            return False, f"OpenRouter error {resp.status_code}: {resp.text[:800]}"
        return False, str(resp)
    try:
        j = resp.json()
        text = _extract_text_like(j)
        if not text:
            # one quick retry with a tiny instruction to force plain text
            payload2 = dict(payload)
            payload2["messages"] = [{"role": "user", "content": f"{prompt}\n\nRespond with plain text only."}]
            r2 = requests.post(OPENROUTER_URL, headers=headers, json=payload2, timeout=timeout)
            if r2.status_code == 200:
                text = _extract_text_like(r2.json())
        return True, (text or "").strip()
    except Exception as e:
        return False, f"openrouter parse error: {e}"


# OpenAI REST (generic)
def ask_openai_rest(prompt: str, timeout: int = OPENAI_TIMEOUT) -> Tuple[bool, str]:
    key = OPENAI_API_KEY
    if not key:
        return False, "openai not configured"

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "400")),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    }

    ok, resp = _do_request_with_retries("post", OPENAI_URL, headers=headers, json_body=body, timeout=timeout)

    if not ok:
        if isinstance(resp, requests.Response):
            return False, f"OpenAI error {resp.status_code}: {resp.text[:800]}"
        return False, str(resp)

    try:
        j = resp.json()
        choices = j.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content") or choices[0].get("text")
            return True, (text or "").strip()
        if "error" in j:
            return False, json.dumps(j["error"])[:800]
        return True, json.dumps(j)[:1000]
    except Exception as e:
        return False, f"OpenAI parse error: {e}"


# DeepSeek wrapper
def ask_deepseek(prompt: str, timeout: int = DEEPSEEK_TIMEOUT) -> Tuple[bool, str]:
    key = DEEPSEEK_API_KEY
    if not key:
        return False, "deepseek not configured"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "400")),
        "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.6"))
    }
    ok, resp = _do_request_with_retries("post", DEEPSEEK_URL, headers=headers, json_body=payload, timeout=timeout)
    if not ok:
        if isinstance(resp, requests.Response):
            return False, f"DeepSeek error {resp.status_code}: {resp.text[:800]}"
        return False, str(resp)
    try:
        j = resp.json()
        choices = j.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content") or choices[0].get("text")
            return True, (text or "").strip()
        # some DeepSeek variants return top-level text
        for k in ("text", "output", "content"):
            if k in j and isinstance(j[k], str):
                return True, j[k].strip()
        return True, json.dumps(j)[:1000]
    except Exception as e:
        return False, f"DeepSeek parse error: {e}"


# Gemini-free wrapper (REST)
def gemini_free_chat(prompt: str, timeout: int = 12) -> tuple[bool, str]:
    api_key = os.getenv("GEMINI_API_KEY")
    base = os.getenv("GEMINI_REST_URL", "https://generativelanguage.googleapis.com/v1beta")
    if not api_key:
        return False, "gemini not configured"
    url = f"{base}/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=body, timeout=timeout)
        if r.status_code != 200:
            return False, f"gemini http {r.status_code}: {r.text[:400]}"
        j = r.json()
        text = j["candidates"][0]["content"]["parts"][0]["text"]
        return True, text.strip()
    except Exception as e:
        return False, f"gemini error: {e}"


# LLaMA local server wrapper (assumes simple JSON API)
def ask_llama(prompt: str, n_predict: int = 256, timeout: int = LLAMA_TIMEOUT) -> Tuple[bool, str]:
    url = LLAMA_SERVER_URL
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt, "n_predict": int(n_predict)}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        return False, f"llama request exception: {e}"

    if r.status_code >= 400:
        return False, f"llama server returned {r.status_code}: {(r.text or '')[:1000]}"
    try:
        j = r.json()
        # try to parse simple shapes
        for k in ("content", "text", "result", "answer", "output", "response"):
            v = j.get(k)
            if isinstance(v, str) and v.strip():
                return True, v.strip()
        choices = j.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                for k in ("text", "content", "message"):
                    v = first.get(k)
                    if isinstance(v, str) and v.strip():
                        return True, v.strip()
        return True, (r.text or "").strip()
    except Exception:
        return True, (r.text or "").strip() or ""


# ---------- Clean FREE_MODELS_PRIORITY parsing ----------
def _split_priority(s: str) -> list[str]:
    # handles stray quotes/whitespace/newlines/commas
    tokens: list[str] = []
    for raw in (s or "").replace("\n", ",").split(","):
        t = raw.strip().strip('"').strip("'")
        if t:
            tokens.append(t)
    return tokens


FREE_MODELS_PRIORITY = _split_priority(os.getenv("FREE_MODELS_PRIORITY", ""))

if not FREE_MODELS_PRIORITY:
    FREE_MODELS_PRIORITY = [
        "openai:gpt-4o-mini",
        "gemini:gemini-2.0-flash",   # matches REST name in your gemini_free_chat()
        "openrouter:gpt-oss-20b",
    ]


# ----------------------
# FREE_MODELS_PRIORITY format: "<provider>:<model>" e.g. "openrouter:gpt-oss-20b"
# ----------------------
def try_free_models_in_order(prompt: str, timeout_each: int = OPENROUTER_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    last_errs: list[str] = []
    for token in FREE_MODELS_PRIORITY:
        prov, model = (token.split(":", 1) + [""])[:2]
        prov, model = prov.strip().lower(), model.strip()

        try:
            if prov in ("openai", "gpt"):
                ok, res = ask_openai_rest(prompt, timeout=timeout_each)
                if ok:
                    return True, res, f"openai/{model or 'default'}"
                last_errs.append(f"openai:{res}")

            elif prov in ("gemini", "google"):
                ok, res = gemini_free_chat(prompt, timeout=timeout_each)
                if ok:
                    return True, res, "gemini-free"
                last_errs.append(f"gemini:{res}")

            elif prov in ("openrouter", "openrouter.ai"):
                if not model:
                    model = "openrouter/auto"
                ok, res = ask_openrouter(model, prompt, timeout=timeout_each)
                if ok:
                    return True, res, f"openrouter/{model}"
                last_errs.append(f"openrouter:{res}")

            elif prov == "llama":
                ok, res = ask_llama(prompt, timeout=timeout_each)
                if ok:
                    return True, res, "llama"
                last_errs.append(f"llama:{res}")

            else:
                last_errs.append(f"{prov}:unknown provider")
        except Exception as e:
            last_errs.append(f"{prov}:{e}")

    return False, "; ".join(last_errs[:5]) or "no free provider configured", None


# --- Initialize SQLite DB on startup ---
# Decide engine once
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db.sqlite3")
IS_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))

# ---- boot-time schema ensure ----
try:
    if IS_PG:
        # Postgres: create/upgrade tables, nothing SQLite-related
        ensure_live_schema_pg()
        print("[hh] init_db skipped on PG ✅")
    else:
        # SQLite: do local file DB setup
        init_db()
        ensure_user_columns()
        ensure_auth_tables()
        ensure_live_schema()
        ensure_schema_migrations()
        ensure_wallet_tables()
        print("[hh] Database initialized ✅ (sqlite)")
except Exception as e:
    # keep this nonfatal, but message should be engine-aware
    print("[hh] init_db failed (nonfatal):", e)


# Allowed origins for browsers
ALLOWED_ORIGINS = {
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://humanhelperai.in",
    "https://www.humanhelperai.in",
    "https://api.humanhelperai.in",  # generally not needed, but ok
    "https://humanhelperai.github.io",
}

import sqlite3 as _sqlite3

if not IS_PG:
    @app.errorhandler(_sqlite3.OperationalError)
    def _sqlite_op_error(e):
        _maybe_fix_schema_from_error(e)
        return jsonify({"ok": False,
                        "error": {"code": "internal_error", "message": "Internal server error"}}), 500

# Use ONE CORS strategy. (A) Flask-Cors:
from flask_cors import CORS
CORS(app, resources={r"*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)


# =======================
# Auth blueprint + routes
# =======================
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


# unified responses
def ok(payload: dict | None = None, status: int = 200):
    return jsonify({"ok": True, **(payload or {})}), status


def err(message: str, status: int = 400, code: str = "bad_request", extra: dict | None = None):
    body = {"ok": False, "error": {"code": code, "message": message}}
    if extra:
        body["error"].update(extra)
    return jsonify(body), status


from werkzeug.exceptions import HTTPException


@app.errorhandler(HTTPException)
def _http_exc(e: HTTPException):
    return err(e.description or "HTTP error", e.code or 500, code=str(e.code))


@app.errorhandler(Exception)
def _uncaught(e: Exception):
    app.logger.exception("uncaught_exception")
    return err("Internal server error", 500, code="internal_error")


def _cors_origin(origin: str) -> str | None:
    if not origin:
        return None
    return origin if origin in ALLOWED_ORIGINS else None


@app.after_request
def add_security_headers(resp):
    # keep your CORS logic above; then:
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    return resp


MOBILE_RE = re.compile(r"^\+?[0-9]{8,15}$")


def require_fields(data: dict, fields: list[str]):
    missing = [f for f in fields if not (data.get(f) or "").strip()]
    if missing:
        return f"Missing fields: {', '.join(missing)}"
    return None


def validate_mobile(m: str) -> bool:
    return bool(MOBILE_RE.match(m or ""))


# Handle preflight quickly
@app.route("/<path:anypath>", methods=["OPTIONS"])
def options_any(anypath):
    return ("", 204)


def _repair_stuck_withdrawals(max_age_min: int = 10, batch_limit: int = 50):
    """
    If a withdrawal sits in 'processing' too long (e.g., Razorpay unreachable),
    release the user's locked funds and mark the request 'failed'.
    """
    db = get_db(); cur = db.cursor()
    try:
        cur.execute("""
            SELECT id, user_id, amount
            FROM withdrawal_requests
            WHERE status='processing'
              AND datetime(created_at) <= datetime('now', ?)
            ORDER BY id ASC
            LIMIT ?
        """, (f'-{max_age_min} minutes', batch_limit))
        stuck = cur.fetchall()
        if not stuck:
            db.close(); return 0

        n = 0
        for r in stuck:
            wid, uid, amount = r["id"], int(r["user_id"]), float(r["amount"])
            try:
                db.execute("BEGIN IMMEDIATE")
                # release lock back to balance
                cur.execute("""
                    UPDATE users
                       SET balance = balance + ?, locked_balance = locked_balance - ?
                     WHERE id=? AND locked_balance >= ?
                """, (amount, amount, uid, amount))
                # ledger
                cur.execute("SELECT balance, locked_balance FROM users WHERE id=?", (uid,))
                b = cur.fetchone()
                bal_after = float(b["balance"] or 0)
                lock_after = float(b["locked_balance"] or 0)
                cur.execute("""
                    INSERT INTO wallet_txns (user_id,kind,amount,balance_after,locked_after,status,ref,note)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (uid, "withdraw_release", +amount, bal_after, lock_after, "failed",
                      f"WREQ:{wid}", "auto-repair: payout stale"))
                # mark failed
                cur.execute("""
                    UPDATE withdrawal_requests
                       SET status='failed', reason='auto-repair: payout stale',
                           updated_at=CURRENT_TIMESTAMP
                     WHERE id=?
                """, (wid,))
                db.commit()
                n += 1
            except Exception as e:
                db.rollback()
                _maybe_fix_schema_from_error(e)
        db.close()
        return n
    except Exception as e:
        db.close()
        _maybe_fix_schema_from_error(e)
        return 0


# light background cleanup thread (non-blocking)
def _cleanup_loop():
    while True:
        try:
            try:
                cleanup_old_logs()
            except Exception:
                pass
            # 🔧 run every cycle
            _repair_stuck_withdrawals(max_age_min=10)
            time.sleep(6 * 60 * 60)
        except Exception as e:
            print("[hh] Cleanup error:", e)
            time.sleep(60)


# start background cleanup thread once
threading.Thread(target=_cleanup_loop, daemon=True).start()


# --- Admin guard (kept - separate from admin blueprint decorator) ---
def admin_required(fn):
    @wraps(fn)
    def _wrap(*a, **k):
        tok = (request.headers.get("X-Admin-Token") or "").strip()
        if not ADMIN_TOKEN:
            return jsonify({"error": "admin not configured"}), 500
        if tok != ADMIN_TOKEN:
            app.logger.warning("admin_required: bad token len=%s", len(tok))
            return jsonify({"error": "admin auth failed"}), 401
        return fn(*a, **k)
    return _wrap


# utilities used by auth
def _utcnow(): return datetime.now(timezone.utc)
def _otp(): return f"{random.randint(100000,999999)}"

MOBILE_IN_RE = re.compile(r'^[6-9]\d{9}$')          # India mobile
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
OTP_TTL_MIN = int(os.getenv("OTP_TTL_MIN", "10"))
SEND_VERIFICATION_MODE = os.getenv("SEND_VERIFICATION_MODE", "console")


def _send_code(dest: str, code: str):
    if SEND_VERIFICATION_MODE == "console":
        print(f"[VERIFY] send code {code} to {dest}")


def _clean(s): return (s or "").strip()















# ----------------------
# Auth: register + verify
# ----------------------

@auth_bp.post("/register")
@limiter.limit("10/hour")
def register():
    d = request.get_json(silent=True) or {}
    name     = _clean(d.get("full_name"))
    mobile   = _clean(d.get("mobile"))
    password = d.get("password") or ""
    email    = _clean(d.get("email"))
    address  = _clean(d.get("address"))

    if not name or len(name) < 2:
        return jsonify({"error": "full_name required"}), 400
    if not MOBILE_IN_RE.match(mobile):
        return jsonify({"error": "invalid mobile (India 10-digit)"}), 400
    if len(password) < 8:
        return jsonify({"error": "password must be at least 8 chars"}), 400
    if email and not EMAIL_RE.match(email):
        return jsonify({"error": "invalid email"}), 400
    if len(address) < 4:
        return jsonify({"error": "address required"}), 400

    code = _otp()
    expires = _now_ts() + OTP_TTL_MIN * 60
    pwd_hash = pbkdf2_sha256.hash(password)

    conn = None
    try:
        conn = get_db()
        # Use db_exec which will normalize %s -> ? for sqlite and keep %s for PG
        sql = """
            INSERT INTO users
               (name,mobile,email,address,password_hash,verification_code,verify_expires,is_verified)
               VALUES (%s,%s,%s,%s,%s,%s,%s,0)
            ON CONFLICT(mobile) DO UPDATE SET
               name = excluded.name,
               email = excluded.email,
               address = excluded.address,
               password_hash = excluded.password_hash,
               verification_code = excluded.verification_code,
               verify_expires = excluded.verify_expires,
               is_verified = 0
        """
        db_exec(conn, sql, (name, mobile, email, address, pwd_hash, code, expires))

        # commit (best-effort; rollback on failure)
        try:
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

        # developer-friendly output + configured sender
        print(f"[VERIFY] send code {code} to {mobile}")
        try:
            _send_code(mobile, code)
        except Exception:
            app.logger.exception("register: _send_code failed")

        return jsonify({
            "message": "verification code sent",
            "mobile": mobile,
            "expires_in_min": OTP_TTL_MIN
        }), 201

    except Exception as e:
        app.logger.exception("register: db error")
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        return jsonify({"ok": False, "error": {"code": "internal_error", "message": str(e)}}), 500

    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


@auth_bp.post("/verify")
@limiter.limit("20/hour")
def verify():
    d = request.get_json(silent=True) or {}
    mobile = _clean(d.get("mobile"))
    code = _clean(d.get("code"))

    if not MOBILE_IN_RE.match(mobile):
        return jsonify({"error": "invalid mobile"}), 400
    if not code.isdigit() or len(code) != 6:
        return jsonify({"error": "invalid code"}), 400

    conn = None
    try:
        conn = get_db()
        # SELECT using engine-agnostic db_exec
        cur = db_exec(conn, "SELECT id, verification_code, verify_expires, is_verified FROM users WHERE mobile=%s", (mobile,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "user not found"}), 404

        # row supports both sqlite3.Row and psycopg2 RealDictRow via mapping access
        is_verified = int(row["is_verified"]) if row["is_verified"] is not None else 0
        if is_verified == 1:
            return jsonify({"message": "already verified"}), 200

        stored_code = (row["verification_code"] or "")
        if stored_code != code:
            return jsonify({"error": "incorrect code"}), 400

        if row["verify_expires"] and _now_ts() > int(row["verify_expires"]):
            return jsonify({"error": "code expired"}), 400

        db_exec(conn, "UPDATE users SET is_verified=1, verification_code=NULL, verify_expires=NULL WHERE id=%s", (row["id"],))
        try:
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

        return jsonify({"message": "verified"}), 200

    except Exception as e:
        app.logger.exception("verify: db error")
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        return jsonify({"ok": False, "error": {"code": "internal_error", "message": str(e)}}), 500
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

@auth_bp.post("/resend-code")
@limiter.limit("5/hour")
def resend_code():
    d = request.get_json(silent=True) or {}
    mobile = _clean(d.get("mobile"))
    if not MOBILE_IN_RE.match(mobile):
        return jsonify({"error": "invalid mobile"}), 400

    db = get_db()
    row = db.execute("SELECT id,is_verified FROM users WHERE mobile=?", (mobile,)).fetchone()
    if not row:
        db.close(); return jsonify({"error": "not registered"}), 404
    if int(row["is_verified"]) == 1:
        db.close(); return jsonify({"message": "already verified"}), 200

    code = _otp()
    expires = _now_ts() + OTP_TTL_MIN * 60
    db.execute("UPDATE users SET verification_code=?, verify_expires=? WHERE id=?", (code, expires, row["id"]))
    db.commit(); db.close()
    _send_code(mobile, code)
    return jsonify({"message": "verification code re-sent", "expires_in_min": OTP_TTL_MIN}), 200


@auth_bp.post("/login")
@limiter.limit("5/minute;50/hour")
def login():
    d = request.get_json(silent=True) or {}
    mobile = _clean(d.get("mobile"))
    password = d.get("password") or ""
    if not MOBILE_IN_RE.match(mobile) or not password:
        return jsonify({"error": "invalid credentials"}), 400

    db = get_db()
    row = db.execute("SELECT id,is_banned,is_verified,password_hash,name,email,address FROM users WHERE mobile=?", (mobile,)).fetchone()
    db.close()

    if not row:
        return jsonify({"error": "user not found"}), 404
    if row["is_banned"]:
        return jsonify({"error": "account banned"}), 403
    if not row["is_verified"]:
        return jsonify({"error": "not verified"}), 403
    if not pbkdf2_sha256.verify(password, row["password_hash"]):
        return jsonify({"error": "invalid credentials"}), 401

    access = make_access_token(row["id"], mobile)
    refresh = make_refresh_token(row["id"])

    return jsonify({
        "ok": True,
        "user": {
            "id": row["id"], "name": row["name"], "mobile": mobile,
            "email": row["email"], "address": row["address"]
        },
        "access": access,
        "refresh": refresh
    }), 200


@auth_bp.post("/refresh")
def refresh():
    """JSON: { refresh }  -> returns new access + rotated refresh"""
    data = request.get_json(silent=True) or {}
    token = (data.get("refresh") or "").strip()
    payload = decode_token(token)
    if not payload or payload.get("type") != "refresh":
        return jsonify({"error": "invalid or expired refresh"}), 401

    jti = payload.get("jti", "")
    conn = get_db()
    row = conn.execute("SELECT user_id, revoked, expires_at FROM refresh_tokens WHERE jti=?", (jti,)).fetchone()
    if (not row) or row["revoked"] or row["expires_at"] < _now_ts():
        conn.close()
        return jsonify({"error": "refresh revoked/expired"}), 401

    # rotate: revoke current, issue new
    conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE jti=?", (jti,))
    conn.commit()

    # fetch mobile for claim
    mrow = conn.execute("SELECT mobile FROM users WHERE id=?", (row["user_id"],)).fetchone()
    conn.close()

    new_access = make_access_token(row["user_id"], mrow["mobile"])
    new_refresh = make_refresh_token(row["user_id"])
    return jsonify({"access": new_access, "refresh": new_refresh}), 200


@auth_bp.post("/logout")
def logout():
    """JSON: { refresh }  -> revoke this refresh token"""
    data = request.get_json(silent=True) or {}
    token = (data.get("refresh") or "").strip()
    payload = decode_token(token)
    if not payload or payload.get("type") != "refresh":
        return jsonify({"error": "invalid refresh"}), 400
    jti = payload.get("jti", "")
    conn = get_db()
    conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE jti=?", (jti,))
    conn.commit()
    conn.close()
    return jsonify({"message": "logged out"}), 200


@app.before_request
def _maybe_purge():
    # ~1% of requests purge old refresh tokens
    if random.randint(1, 100) == 1:
        try:
            clean_expired_refresh()
        except Exception:
            pass


@app.get("/whoami")
@auth_required
def whoami():
    return jsonify({"user_id": g.user_id, "mobile": g.mobile})


# register blueprint once
app.register_blueprint(auth_bp)









# ----------------------
# Simple utility endpoints
# ----------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Human Helper API is live", "status": "ok"}), 200


# Optional: don’t rate-limit health checks
@limiter.exempt
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"message": "Human Helper API is live", "status": "ok"}), 200


# ✅ Simple echo test route
@app.route("/echo", methods=["POST"])
def echo():
    return jsonify({"you_sent": request.json or {}})


@app.route("/version", methods=["GET"])
def version():
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("COMMIT_SHA", "dev")
    return jsonify({"commit": sha}), 200


@app.get("/debug/whichdb")
def debug_whichdb():
    if IS_PG:
        info = {"engine": "postgres", "url_prefix": DATABASE_URL[:25] + "..."}
        try:
            db = get_db()
            row = db.execute("SELECT version() AS v").fetchone()
            info["version"] = row["v"] if row else None
            db.close()
        except Exception as e:
            info["error"] = str(e)
        return jsonify(info)
    else:
        db_file = DATABASE_URL.replace("sqlite:///", "")
        return jsonify({"engine": "sqlite", "db_file": db_file, "exists": os.path.exists(db_file)})










# ----------------------
# AI endpoints: ai/ask (simple), ai/humanhelper (routing with free-first + premium fallback)
# ----------------------
@app.route("/ai/ask", methods=["POST"])
def ai_ask():
    """
    Lightweight single-call AI endpoint (tries FREE_MODELS_PRIORITY in order).
    """
    data = request.get_json(force=False) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "prompt required"}), 400

    ok, ans_or_err, provider = try_free_models_in_order(prompt, timeout_each=10)
    if ok:
        return jsonify({"ok": True, "provider": provider or "free", "answer": ans_or_err}), 200

    # Premium users get a deterministic paid-first pass: OpenAI -> Gemini -> OpenRouter -> LLaMA
    if is_premium_user(request):
        # 1) OpenAI
        if OPENAI_API_KEY:
            ok1, res1 = ask_openai_rest(prompt, timeout=OPENAI_TIMEOUT)
            if ok1:
                return jsonify({"ok": True, "provider": "openai", "answer": res1}), 200

        # 2) Gemini
        ok2, res2 = gemini_free_chat(prompt, timeout=GEMINI_TIMEOUT)
        if ok2:
            return jsonify({"ok": True, "provider": "gemini", "answer": res2}), 200

        # 3) OpenRouter
        if OPENROUTER_API_KEY:
            # try first openrouter-* token in priority list if present
            for token in FREE_MODELS_PRIORITY:
                if token.startswith("openrouter"):
                    model = token.split(":", 1)[1] if ":" in token else "gpt-oss-20b"
                    ok3, res3 = ask_openrouter(model, prompt, timeout=OPENROUTER_TIMEOUT)
                    if ok3:
                        return jsonify({"ok": True, "provider": f"openrouter/{model}", "answer": res3}), 200
                    break

        # 4) LLaMA
        ok4, res4 = ask_llama(prompt, timeout=LLAMA_TIMEOUT)
        if ok4:
            return jsonify({"ok": True, "provider": "llama", "answer": res4}), 200

    return jsonify({"ok": False, "provider": "none", "error": ans_or_err}), 502


@app.route("/ai/humanhelper", methods=["POST"])
def ai_humanhelper():
    """
    Rich routing with conversation history support in the future.
    For now, mirrors ai/ask logic but keeps premium paid-first path.
    """
    data = request.get_json(force=False) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "prompt required"}), 400

    premium = is_premium_user(request)

    if not premium:
        ok, ans, prov = try_free_models_in_order(prompt, timeout_each=10)
        if ok:
            return jsonify({"ok": True, "provider": prov or "free", "answer": ans}), 200
        llama_ok, llama_ans = ask_llama(prompt, timeout=LLAMA_TIMEOUT)
        if llama_ok:
            return jsonify({"ok": True, "provider": "llama", "answer": llama_ans}), 200
        return jsonify({"ok": False, "provider": "none", "error": ans}), 502

    # premium path: OpenAI -> Gemini -> OpenRouter -> LLaMA
    if OPENAI_API_KEY:
        ok_oa, ans_oa = ask_openai_rest(prompt, timeout=OPENAI_TIMEOUT)
        if ok_oa:
            return jsonify({"ok": True, "provider": "openai", "answer": ans_oa}), 200

    ok_g, ans_g = gemini_free_chat(prompt, timeout=GEMINI_TIMEOUT)
    if ok_g:
        return jsonify({"ok": True, "provider": "gemini", "answer": ans_g}), 200

    if OPENROUTER_API_KEY:
        for tok in FREE_MODELS_PRIORITY:
            if tok.startswith("openrouter"):
                model = tok.split(":", 1)[1] if ":" in tok else "gpt-oss-20b"
                ok_or, res_or = ask_openrouter(model, prompt, timeout=OPENROUTER_TIMEOUT)
                if ok_or:
                    return jsonify({"ok": True, "provider": f"openrouter/{model}", "answer": res_or}), 200
                break

    llama_ok, llama_ans = ask_llama(prompt, timeout=LLAMA_TIMEOUT)
    if llama_ok:
        return jsonify({"ok": True, "provider": "llama", "answer": llama_ans}), 200

    return jsonify({"ok": False, "provider": "none", "error": "All providers failed."}), 502


@app.route("/ai/status", methods=["GET"])
def ai_status():
    return jsonify({
        "openai_configured": bool(OPENAI_API_KEY),
        "gemini_configured": bool(GEMINI_REST_URL and GEMINI_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "llama_server": LLAMA_SERVER_URL,
        "free_models_priority": FREE_MODELS_PRIORITY,
        "notes": {
            "free_first": "Tries providers exactly in FREE_MODELS_PRIORITY order.",
            "premium_flow": "OpenAI -> Gemini -> OpenRouter -> LLaMA."
        }
    })









# =========================
# WALLET BLUEPRINT + ROUTES
# =========================

wallet_bp = Blueprint("wallet", __name__, url_prefix="/wallet")

MIN_WITHDRAW = 100.0          # ₹100
FEE_RATE     = 0.08           # 8% total -> goes to ONE pool

def _upi_validate(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.match(r"^[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}$", s))

def _qr_payload_for_user(user_id: int) -> str:
    return f"hhpay:user:{user_id}"

def _calc_fee(amount: float) -> float:
    return round(amount * FEE_RATE, 2)

@wallet_bp.get("/balance")
@limiter.limit("20/minute")
@auth_required
def wallet_balance():
    db = get_db()
    row = db.execute("SELECT balance, locked_balance FROM users WHERE id=?", (g.user_id,)).fetchone()
    db.close()
    if not row:
        return err("user not found", 404)
    bal = float(row["balance"] or 0)
    locked = float(row["locked_balance"] or 0)
    return ok({"balance": round(bal,2), "locked": round(locked,2), "available": round(bal-locked,2)})

@wallet_bp.get("/me/qr")
@limiter.limit("30/minute")
@auth_required
def wallet_my_qr():
    return ok({"qr_payload": _qr_payload_for_user(g.user_id)})

@wallet_bp.get("/transactions")
@limiter.limit("30/minute")
@auth_required
def wallet_transactions():
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    offset = max(int(request.args.get("offset", 0)), 0)
    db = get_db()
    rows = db.execute("""
        SELECT id, kind, amount, balance_after, locked_after, status, ref, note, meta, created_at
        FROM wallet_txns
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """, (g.user_id, limit, offset)).fetchall()
    db.close()
    txns = []
    for r in rows:
        txns.append({
            "id": r["id"],
            "kind": r["kind"],
            "amount": float(r["amount"]),
            "balance_after": float(r["balance_after"]),
            "locked_after": float(r["locked_after"]),
            "status": r["status"],
            "ref": r["ref"],
            "note": r["note"],
            "meta": json.loads(r["meta"]) if r["meta"] else None,
            "created_at": r["created_at"],
        })
    return ok({"txns": txns})

@wallet_bp.post("/transfer")
@limiter.limit("20/minute")
@auth_required
def wallet_transfer():
    d = request.get_json(silent=True) or {}
    amount = float(d.get("amount") or 0)
    receiver_id = d.get("receiver_id")
    qr = (d.get("qr_payload") or "").strip()

    if amount <= 0:
        return err("amount must be > 0", 400)
    if not receiver_id and qr:
        m = re.match(r"^hhpay:user:(\d+)$", qr)
        if not m:
            return err("invalid qr payload", 400)
        receiver_id = int(m.group(1))
    try:
        receiver_id = int(receiver_id)
    except Exception:
        return err("receiver_id required", 400)
    if receiver_id == g.user_id:
        return err("cannot transfer to self", 400)

    db = get_db(); cur = db.cursor()
    try:
        db.execute("BEGIN IMMEDIATE")
        s = db_exec(cur, "SELECT balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        if not s:
            raise ValueError("sender not found")
        if float(s["balance"]) < amount:
            raise ValueError("insufficient balance")
        if not db_exec(cur, "SELECT 1 FROM users WHERE id=?", (receiver_id,)).fetchone():
            raise ValueError("receiver not found")

        db_exec(cur, "UPDATE users SET balance = balance - ? WHERE id=?", (amount, g.user_id))
        r1 = db_exec(cur, "SELECT balance, locked_balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        db_exec(cur, """INSERT INTO wallet_txns
            (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
            VALUES (?,?,?,?,?,'success',NULL,?,NULL)
        """, (g.user_id, "p2p_out", -amount, r1["balance"], r1["locked_balance"], f"to user:{receiver_id}"))

        db_exec(cur, "UPDATE users SET balance = balance + ? WHERE id=?", (amount, receiver_id))
        r2 = db_exec(cur, "SELECT balance, locked_balance FROM users WHERE id=?", (receiver_id,)).fetchone()
        db_exec(cur, """INSERT INTO wallet_txns
            (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
            VALUES (?,?,?,?,?,'success',NULL,?,NULL)
        """, (receiver_id, "p2p_in", amount, r2["balance"], r2["locked_balance"], f"from user:{g.user_id}"))

        db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        db.close()
        return err(str(e), 400)
    db.close()
    return ok({"message":"transfer complete"})

@wallet_bp.post("/withdraw")
@limiter.limit("5/hour")
@auth_required
def wallet_withdraw():
    d = request.get_json(silent=True) or {}
    amount = float(d.get("amount") or 0)
    upi_id = (d.get("upi_id") or "").strip()

    if amount < MIN_WITHDRAW:
        return err(f"minimum withdrawal is ₹{int(MIN_WITHDRAW)}", 400)
    if not _upi_validate(upi_id):
        return err("invalid UPI id", 400)

    fee_total = _calc_fee(amount)
    net = round(amount - fee_total, 2)
    if net <= 0:
        return err("amount too low after fees", 400)

    db = get_db(); cur = db.cursor()
    wreq_id = None; payout_id = None
    try:
        db.execute("BEGIN IMMEDIATE")
        r = db_exec(cur, "SELECT balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        if not r or float(r["balance"]) < amount:
            raise ValueError("insufficient balance")

        # move to locked
        db_exec(cur, """
            UPDATE users
            SET balance = balance - ?, locked_balance = locked_balance + ?
            WHERE id=?
        """, (amount, amount, g.user_id))
        rlock = db_exec(cur, "SELECT balance, locked_balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        db_exec(cur, """INSERT INTO wallet_txns
            (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
            VALUES (?,?,?,?,?,'requested',?, ?, ?)
        """, (g.user_id, "withdraw_lock", -amount, rlock["balance"], rlock["locked_balance"],
              None, f"lock for withdraw {amount}",
              json.dumps({"upi": upi_id, "net": net, "fee": fee_total})))

        # record withdrawal request as processing
        db_exec(cur, """INSERT INTO withdrawal_requests
            (user_id, amount, net_amount, fee_amount, upi, status)
            VALUES (?,?,?,?,?,'processing')""", (g.user_id, amount, net, fee_total, upi_id))
        wreq_id = cur.lastrowid
        db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        db.close()
        return err(str(e), 400)

    # call Razorpay outside transaction
    ok_pay, res = _razorpay_payout_upi(upi_id, net, ref=f"WREQ:{wreq_id}")

    db = get_db(); cur = db.cursor()
    try:
        db.execute("BEGIN IMMEDIATE")
        if not ok_pay:
            # release lock
            db_exec(cur, """
                UPDATE users
                SET balance = balance + ?, locked_balance = locked_balance - ?
                WHERE id=?
            """, (amount, amount, g.user_id))
            rrel = db_exec(cur, "SELECT balance, locked_balance FROM users WHERE id=?", (g.user_id,)).fetchone()
            db_exec(cur, """INSERT INTO wallet_txns
                (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
                VALUES (?,?,?,?,?,'failed',?, ?, ?)""",
                (g.user_id, "withdraw_release", amount, rrel["balance"], rrel["locked_balance"],
                 f"WREQ:{wreq_id}", "payout failed", json.dumps({"err": str(res)[:240]})))
            db_exec(cur, "UPDATE withdrawal_requests SET status='failed', reason=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (str(res)[:240], wreq_id))
            db.commit(); db.close()
            return err("payout failed", 502, extra={"details": str(res)})

        # success: clear lock, credit pooled fee, mark paid
        payout_id = res.get("id") if isinstance(res, dict) else None
        db_exec(cur, "UPDATE users SET locked_balance = locked_balance - ? WHERE id=?", (amount, g.user_id))
        rset = db_exec(cur, "SELECT balance, locked_balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        db_exec(cur, """INSERT INTO wallet_txns
            (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
            VALUES (?,?,?,?,?,'success',?, ?, NULL)""",
            (g.user_id, "withdraw_settle", 0.0, rset["balance"], rset["locked_balance"],
             f"WREQ:{wreq_id}", f"payout success: {payout_id}"))

        # add the ENTIRE fee to the single pool
        db_exec(cur, """
            UPDATE fee_pool
            SET pool_balance = pool_balance + ?, updated_at = CURRENT_TIMESTAMP
            WHERE id=1
        """, (fee_total,))

        db_exec(cur, "UPDATE withdrawal_requests SET status='paid', payout_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (payout_id, wreq_id))
        db.commit()
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        db.close()
        return err("post-payout update failed", 500, extra={"details": str(e)})
    db.close()

    return ok({
        "message": "withdrawal processed",
        "request_id": wreq_id,
        "net_paid": net,
        "fee": fee_total,
        "payout_id": payout_id
    })


# ----------------------
# Wallet helpers (module-level)
# ----------------------
def _user_by_id(uid: int):
    db = get_db()
    cur = db.cursor()
    row = db_exec(cur, "SELECT id, balance, locked_balance, is_verified, is_banned FROM users WHERE id=?", (uid,)).fetchone()
    db.close()
    return row


try:
    from writer import start_writer, enqueue_write
except Exception:
    def start_writer():
        print("[hh] writer.start_writer stub")

    def enqueue_write(sql, params=(), timeout=1.0):
        # emulate synchronous success
        return True, None, ["immediate"], 0, None

    print("[hh] Warning: writer module not found — using stub enqueue_write")


try:
    from wallet import deposit, withdraw
except Exception:
    def deposit(mobile, amount):
        return False, "wallet not configured"
    def withdraw(mobile, amount):
        return False, "wallet not configured"
    print("[hh] Warning: wallet module not found — using stubs")


try:
    from earnings import reward_user
except Exception:
    def reward_user(mobile, video_id, content_type, duration):
        return False, "earnings not configured"
    print("[hh] Warning: earnings module not found — using stubs")


# -------- Razorpay Payout (UPI only) --------
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_PAYOUT_URL = "https://api.razorpay.com/v1/payouts"

def _razorpay_payout_upi(upi_id: str, amount_inr: float, ref: str, timeout: int = 12) -> tuple[bool, dict | str]:
    if not (RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET):
        return False, "razorpay not configured"

    # amount in paise
    amt = int(round(amount_inr * 100))
    payload = {
        "account_number": os.getenv("RAZORPAY_VPA_SOURCE_ACC", ""),  # Your virtual account/RAZORPAY account no
        "fund_account": {
            "account_type": "vpa",
            "vpa": {"address": upi_id},
            "contact": {"name": "HH User", "type": "employee"}  # Razorpay requires a contact; VPA payouts allow inline
        },
        "amount": amt,
        "currency": "INR",
        "mode": "UPI",
        "purpose": "payout",
        "queue_if_low_balance": True,
        "reference_id": ref,
        "narration": "HumanHelper Withdrawal"
    }
    try:
        r = requests.post(
            RAZORPAY_PAYOUT_URL,
            auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
            json=payload, timeout=timeout
        )
        if r.status_code >= 400:
            return False, f"razorpay {r.status_code}: {(r.text or '')[:500]}"
        return True, r.json()
    except Exception as e:
        return False, str(e)


# register wallet blueprint once
app.register_blueprint(wallet_bp)








# ----------------------
# GitHub helpers
# ----------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

def github_get_file(owner: str, repo: str, path: str, ref: str = "main"):
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not configured"
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw"
    }
    ok, resp = _do_request_with_retries("get", url, headers=headers, timeout=10)
    if not ok:
        return False, f"GitHub request failed: {resp}"
    return True, resp.text

def github_create_gist(files: dict, description: str = "", public: bool = False):
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not configured"
    url = "https://api.github.com/gists"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "files": {name: {"content": content} for name, content in files.items()},
        "description": description,
        "public": public
    }
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=10)
    if not ok:
        return False, f"GitHub gist creation failed: {resp}"
    try:
        j = resp.json()
        return True, j.get("html_url") or j.get("url")
    except Exception:
        return False, "GitHub gist parse error"

# Public read (consider keeping it admin-only if you prefer)
@limiter.limit("20/minute")
@app.route("/github/get", methods=["GET"])
def github_get():
    owner = request.args.get("owner")
    repo = request.args.get("repo")
    path = request.args.get("path")
    ref = request.args.get("ref", "main")
    if not (owner and repo and path):
        return jsonify({"error": "owner,repo,path required"}), 400
    ok, res = github_get_file(owner, repo, path, ref=ref)
    if not ok:
        return jsonify({"error": res}), 400
    return jsonify({"ok": True, "content": res})

# Admin-only create gist
@limiter.limit("10/minute")
@app.route("/github/gist", methods=["POST"])
@admin_required
def github_gist():
    d = request.json or {}
    files = d.get("files")
    desc = d.get("description", "")
    public = bool(d.get("public", False))
    if not files or not isinstance(files, dict):
        return jsonify({"error": "files dict required"}), 400
    ok, res = github_create_gist(files, description=desc, public=public)
    if not ok:
        return jsonify({"error": res}), 400
    return jsonify({"ok": True, "url": res})









# ----------------------
# Start-up: init DB and writer non-fatally
# ----------------------
try:
    if not IS_PG:
        init_db()
except Exception as e:
    print("[hh] init_db failed (nonfatal):", e)

try:
    start_writer()
except Exception as e:
    print("[hh] start_writer failed (nonfatal):", e)








# ----------------------
# Run app
# ----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("[hh] Starting HumanHelper backend on port", port)
    print("[hh] OPENROUTER configured:", bool(OPENROUTER_API_KEY and OPENROUTER_URL))
    print("[hh] OpenAI configured:", bool(OPENAI_API_KEY))
    print("[hh] DEEPSEEK configured:", bool(DEEPSEEK_API_KEY))
    print("[hh] FREE_MODELS_PRIORITY:", FREE_MODELS_PRIORITY)
    app.run(host="0.0.0.0", port=port, debug=True)



