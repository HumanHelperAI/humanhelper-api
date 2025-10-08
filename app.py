#!/usr/bin/env python3
# app.py - consolidated HumanHelper backend (all-in-one, defensive)

import os
import time
import json
import random
import threading
from typing import Tuple, Optional, Any, Dict, List
from functools import wraps

import requests
from flask import Flask, request, jsonify, Blueprint, g
import jwt, secrets, uuid
from datetime import datetime, timedelta, timezone
from flask_cors import CORS
import re, json, logging 
from werkzeug.exceptions import HTTPException
from passlib.hash import pbkdf2_sha256


# --- SQLite helper for Termux/mobile ---
import sqlite3
DB_PATH = os.getenv("DATABASE_URL", "db.sqlite3")
# normalize common forms like sqlite:///db.sqlite3
if DB_PATH.startswith("sqlite:///"):
    DB_PATH = DB_PATH[len("sqlite:///"):]
if DB_PATH.startswith("file:"):
    DB_PATH = DB_PATH[len("file:"):]

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ✅ Ensure users table and columns exist before app starts
def ensure_user_columns():
    conn = get_db()
    cur = conn.cursor()

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
            is_banned INTEGER DEFAULT 0,
            balance REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add any missing columns defensively
    cur.execute("PRAGMA table_info(users)")
    have = {row["name"] for row in cur.fetchall()}

    wanted = {
        "name": "TEXT",
        "mobile": "TEXT",
        "email": "TEXT",
        "address": "TEXT",
        "password_hash": "TEXT",
        "verification_code": "TEXT",
        "verify_expires": "INTEGER",                 # OTP expiry (epoch seconds)
        "is_verified": "INTEGER DEFAULT 0",
        "is_banned": "INTEGER DEFAULT 0",
        "balance": "REAL DEFAULT 0",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    }
    for col, typ in wanted.items():
        if col not in have:
            cur.execute(f"ALTER TABLE users ADD COLUMN {col} {typ}")

    conn.commit()
    conn.close()

def ensure_auth_tables():
    """Create refresh token table and indexes if missing (idempotent)."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens(
            jti TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            issued_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            revoked INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_exp  ON refresh_tokens(expires_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_refresh_rev  ON refresh_tokens(revoked)")
    conn.commit()
    conn.close()

def ensure_schema_migrations():
    db = get_db()
    cur = db.cursor()

    # ---- users: add missing columns ----
    cur.execute("PRAGMA table_info(users)")
    cols = {r[1] for r in cur.fetchall()}
    if "email" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if "address" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN address TEXT")
    if "locked_balance" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN locked_balance REAL DEFAULT 0")
    if "is_verified" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
    if "is_banned" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN is_banned INTEGER DEFAULT 0")

    # ---- wallet_txns (immutable ledger) ----
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")

    # ---- withdrawal_requests ----
    cur.execute("""
    CREATE TABLE IF NOT EXISTS withdrawal_requests(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        amount REAL NOT NULL,
        net_amount REAL NOT NULL,
        fee_amount REAL NOT NULL,
        upi TEXT,
        payout_id TEXT,
        status TEXT NOT NULL DEFAULT 'requested',
        reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")

    # ---- fee_pool: migrate old -> new single 'balance' column ----
    # detect current fee_pool shape
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fee_pool'")
    has_fee = cur.fetchone() is not None
    if not has_fee:
        cur.execute("""
        CREATE TABLE fee_pool (
            id INTEGER PRIMARY KEY CHECK (id=1),
            balance REAL NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        cur.execute("INSERT OR IGNORE INTO fee_pool (id, balance) VALUES (1, 0)")
    else:
        cur.execute("PRAGMA table_info(fee_pool)")
        fcols = {r[1] for r in cur.fetchall()}
        if {"charity_balance", "maintenance_balance"}.issubset(fcols) and "balance" not in fcols:
            # read old totals
            cur.execute("SELECT COALESCE(charity_balance,0), COALESCE(maintenance_balance,0) FROM fee_pool WHERE id=1")
            row = cur.fetchone()
            total = float((row[0] if row else 0) + (row[1] if row else 0))
            # migrate: create new table, copy total, drop old
            cur.execute("ALTER TABLE fee_pool RENAME TO fee_pool_old")
            cur.execute("""
            CREATE TABLE fee_pool (
                id INTEGER PRIMARY KEY CHECK (id=1),
                balance REAL NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            cur.execute("INSERT OR REPLACE INTO fee_pool (id, balance) VALUES (1, ?)", (total,))
            cur.execute("DROP TABLE fee_pool_old")
        elif "balance" not in fcols:
            # unexpected shape -> recreate safely with zero
            cur.execute("DROP TABLE IF EXISTS fee_pool")
            cur.execute("""
            CREATE TABLE fee_pool (
                id INTEGER PRIMARY KEY CHECK (id=1),
                balance REAL NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            cur.execute("INSERT OR IGNORE INTO fee_pool (id, balance) VALUES (1, 0)")

    db.commit()
    db.close()

# --- Wallet schema (single pooled fee) ---
def ensure_wallet_tables():
    conn = get_db(); cur = conn.cursor()

    # users.locked_balance if missing
    cur.execute("PRAGMA table_info(users)")
    have = {r["name"] for r in cur.fetchall()}
    if "locked_balance" not in have:
        cur.execute("ALTER TABLE users ADD COLUMN locked_balance REAL DEFAULT 0")

    # immutable ledger
    cur.execute("""
    CREATE TABLE IF NOT EXISTS wallet_txns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        kind TEXT NOT NULL,        -- deposit|earn|withdraw_lock|withdraw_settle|withdraw_release|fee_pool|p2p_out|p2p_in
        amount REAL NOT NULL,      -- +credit/-debit from user's perspective
        balance_after REAL NOT NULL,
        locked_after REAL NOT NULL,
        status TEXT NOT NULL,      -- requested|success|failed|rejected
        ref TEXT,
        note TEXT,
        meta TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")

    # withdrawal requests
    cur.execute("""
    CREATE TABLE IF NOT EXISTS withdrawal_requests(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        amount REAL NOT NULL,       -- gross amount user requested
        net_amount REAL NOT NULL,   -- amount paid to user after fee
        fee_amount REAL NOT NULL,   -- total 8% fee (goes to pool)
        upi TEXT,
        payout_id TEXT,
        status TEXT NOT NULL DEFAULT 'requested',  -- requested|processing|paid|failed
        reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")

    # single pooled fee bucket
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fee_pool (
        id INTEGER PRIMARY KEY CHECK (id=1),
        pool_balance REAL NOT NULL DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("INSERT OR IGNORE INTO fee_pool (id,pool_balance) VALUES (1,0)")

    conn.commit(); conn.close()



# ----------------------                                                              
# Flask app init
# ----------------------
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded

app = Flask(__name__)

app.url_map.strict_slashes = False

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
    return jsonify({"ok": False, "error": {"code":"rate_limited","message":"Too many requests. Try later."}}), 429



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

MIN_WITHDRAW = 100.0          # ₹100
FEE_RATE = 0.08               # 8%
FEE_SPLIT = (0.5, 0.5)        # 50/50 => charity, maintenance

def _fetch_user(uid: int):
    db = get_db()
    row = db.execute("SELECT id,balance,locked_balance,is_verified,is_banned FROM users WHERE id=?", (uid,)).fetchone()
    db.close()
    return row

def _insert_ledger(
    db, user_id: int, kind: str, delta_amount: float,
    balance_after: float, locked_after: float,
    status: str, ref: str | None = None, note: str | None = None, meta: dict | None = None
):
    db.execute("""
        INSERT INTO wallet_txns (
            user_id, kind, amount, balance_after, locked_after, status, ref, note, meta
        ) VALUES (?,?,?,?,?,?,?,?,?)
    """, (user_id, kind, delta_amount, balance_after, locked_after, status, ref, note,
          json.dumps(meta) if meta is not None else None))

def _calc_fee(amount: float) -> tuple[float,float,float]:
    """Returns (fee_total, fee_charity, fee_maint)."""
    fee = round(amount * FEE_RATE, 2)
    ch = round(fee * FEE_SPLIT[0], 2)
    mt = round(fee - ch, 2)
    return fee, ch, mt

def _upi_validate(s: str) -> bool:
    s = (s or "").strip()
    # simple UPI validation: letters/digits/.-_ + @ + handle
    return bool(re.match(r"^[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}$", s))

def _qr_payload_for_user(user_id: int) -> str:
    # You can display this string as QR on frontend; payer scans & posts to /wallet/transfer with receiver_id
    return f"hhpay:user:{user_id}"

def cleanup_old_logs():
        # nothing to do for SQLite demo
        pass



# =========================                                                                                                      
# ADMIN BLUEPRINT + ROUTES
# =========================

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

def admin_required(fn):
    @wraps(fn)
    def _wrap(*a, **k):
        token = request.headers.get("X-Admin-Token", "")
        if not token or token != os.getenv("ADMIN_TOKEN", "changeme"):
            return jsonify({"ok": False, "error": {"code": "forbidden", "message": "admin auth failed"}}), 403
        return fn(*a, **k)
    return _wrap


# ----- Fee pool (read-only dashboard) -----
@admin_bp.get("/wallet/fees")
@admin_required
def admin_fees_read():
    db = get_db()
    row = db.execute("SELECT charity_balance, maintenance_balance, updated_at FROM fee_pool WHERE id=1").fetchone()
    db.close()
    if not row:
        return jsonify({"ok": False, "error": {"code": "not_found", "message": "fee pool not initialized"}}), 404
    return jsonify({
        "ok": True,
        "charity": float(row["charity_balance"]),
        "maintenance": float(row["maintenance_balance"]),
        "updated_at": row["updated_at"]
    })


# ----- Fee pool transfer (optional) -----
# Use this to periodically clear fee_pool into your accounting system.
@admin_bp.post("/wallet/fees/transfer")
@admin_required
def admin_fees_transfer():
    d = request.get_json(silent=True) or {}
    from_pool = (d.get("from") or "").strip().lower()          # "charity" or "maintenance"
    amount = float(d.get("amount") or 0)
    note = (d.get("note") or "").strip()

    if from_pool not in ("charity", "maintenance"):
        return jsonify({"ok": False, "error": {"code": "bad_request", "message": "from must be charity|maintenance"}}), 400
    if amount <= 0:
        return jsonify({"ok": False, "error": {"code": "bad_request", "message": "amount must be > 0"}}), 400

    col = "charity_balance" if from_pool == "charity" else "maintenance_balance"
    db = get_db(); cur = db.cursor()
    try:
        db.execute("BEGIN IMMEDIATE")
        r = cur.execute(f"SELECT {col} FROM fee_pool WHERE id=1").fetchone()
        if not r or float(r[0]) < amount:
            raise ValueError("insufficient pool balance")
        # reduce pool
        cur.execute(f"UPDATE fee_pool SET {col} = {col} - ?, updated_at=CURRENT_TIMESTAMP WHERE id=1", (amount,))
        # audit it into a synthetic admin ledger row (user_id=0)
        cur.execute("""
            INSERT INTO wallet_txns (user_id, kind, amount, balance_after, locked_after, status, ref, note, meta)
            VALUES (0, ?, ?, 0, 0, 'success', ?, ?, ?)
        """, (f"fee_transfer_{from_pool}", -amount, f"FEEPOOL:{from_pool}", note, json.dumps({"by": "admin"})))
        db.commit()
    except Exception as e:
        db.rollback(); db.close()
        return jsonify({"ok": False, "error": {"code": "failed", "message": str(e)}}), 400
    db.close()
    return jsonify({"ok": True, "message": f"transferred {amount} from {from_pool} pool"})


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
    cur.execute("UPDATE users SET is_banned=1 WHERE mobile=?", (mobile,))
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
    cur.execute("UPDATE users SET is_banned=0 WHERE mobile=?", (mobile,))
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
    status = (request.args.get("status") or "").strip().lower()  # requested|processing|paid|failed (optional)
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
    cur.execute("UPDATE users SET balance = balance + ? WHERE id=?", (delta, user_id))
    db.commit()
    db.close()

    return jsonify({"ok": True, "message": f"Adjusted ₹{delta} for user {user_id}", "note": note})

# ✅ Register admin blueprint once (AFTER all routes above are defined)
app.register_blueprint(admin_bp)




# ----------------------
# Wallet helpers (module-level)                                                                                                   
# ----------------------
def _user_by_id(uid: int):
    db = get_db()
    row = db.execute(
        "SELECT id, balance, locked_balance, is_verified, is_banned FROM users WHERE id=?",
        (uid,)
    ).fetchone()
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
uyRAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _now() -> int: return int(time.time())
def _make_jti() -> str: return uuid.uuid4().hex

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-in-railway")
JWT_ALGO   = "HS256"
ACCESS_TTL_MIN   = int(os.getenv("ACCESS_TTL_MIN", "30"))
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TTL_DAYS", "30"))

def _utcnow(): return datetime.now(timezone.utc)
def _now_ts(): return int(_utcnow().timestamp())

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

    conn = get_db()
    conn.execute(
        "INSERT INTO refresh_tokens (jti,user_id,issued_at,expires_at,revoked) VALUES (?,?,?,?,0)",
        (jti, user_id, int(now.timestamp()), int(exp.timestamp()))
    )
    conn.commit(); conn.close()
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
            return jsonify({"error":"missing token"}), 401
        payload = decode_token(auth.split(" ",1)[1])
        if not payload or payload.get("type") != "access":
            return jsonify({"error":"invalid or expired token"}), 401
        g.user_id = int(payload["sub"])
        g.mobile = payload.get("mobile") or ""
        return fn(*a, **k)
    return _wrap

def clean_expired_refresh():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM refresh_tokens WHERE expires_at < ?", (_now(),))
    conn.commit()
    conn.close()
# ==== /JWT ====================================================================



# ----------------------
# Utilities: HTTP with retries/backoff                                                                                            
# ----------------------
def _do_request_with_retries(method: str, url: str, headers: Optional[dict] = None, json_body: Optional[dict] = None,
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
                        time.sleep(float(ra) + random.uniform(0.1,0.6))
                    except Exception:
                        time.sleep(backoff_base * (2 ** attempt) + random.uniform(0,0.3))
                else:
                    time.sleep(backoff_base * (2 ** attempt) + random.uniform(0,0.3))
                attempt += 1
                continue
            return False, r
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(backoff_base * (2 ** attempt) + random.uniform(0,0.3))
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
            payload2["messages"] = [{"role":"user","content": f"{prompt}\n\nRespond with plain text only."}]
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

    # ✅ Make sure this line is indented **4 spaces** under the function
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
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS","400")),
        "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE","0.6"))
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
def gemini_free_chat(prompt: str, timeout: int = 12) -> tuple[bool,str]:
    api_key = os.getenv("GEMINI_API_KEY")
    base = os.getenv("GEMINI_REST_URL","https://generativelanguage.googleapis.com/v1beta")
    if not api_key: 
        return False, "gemini not configured"
    url = f"{base}/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = {"contents":[{"parts":[{"text":prompt}]}]}
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
def ask_llama(prompt: str, n_predict: int = 256, timeout: int = LLAMA_TIMEOUT) -> Tuple[bool,str]:
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
        for k in ("content","text","result","answer","output","response"):
            v = j.get(k)
            if isinstance(v, str) and v.strip():
                return True, v.strip()
        choices = j.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                for k in ("text","content","message"):
                    v = first.get(k)
                    if isinstance(v, str) and v.strip():
                        return True, v.strip()
        return True, (r.text or "").strip()
    except Exception:
        return True, (r.text or "").strip() or ""

# ---------- Clean FREE_MODELS_PRIORITY parsing ----------
def _split_priority(s: str) -> list[str]:
    # handles stray quotes/whitespace/newlines/commas
    tokens = []
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
    last_errs = []
    for token in FREE_MODELS_PRIORITY:
        prov, model = (token.split(":", 1) + [""])[:2]
        prov, model = prov.strip().lower(), model.strip()

        try:
            if prov in ("openai", "gpt"):
                ok, res = ask_openai_rest(prompt, timeout=timeout_each)
                if ok: return True, res, f"openai/{model or 'default'}"
                last_errs.append(f"openai:{res}")

            elif prov in ("gemini", "google"):
                ok, res = gemini_free_chat(prompt, timeout=timeout_each)
                if ok: return True, res, "gemini-free"
                last_errs.append(f"gemini:{res}")

            elif prov in ("openrouter", "openrouter.ai"):
                if not model: model = "openrouter/auto"
                ok, res = ask_openrouter(model, prompt, timeout=timeout_each)
                if ok: return True, res, f"openrouter/{model}"
                last_errs.append(f"openrouter:{res}")

            elif prov == "llama":
                ok, res = ask_llama(prompt, timeout=timeout_each)
                if ok: return True, res, "llama"
                last_errs.append(f"llama:{res}")

            else:
                last_errs.append(f"{prov}:unknown provider")
        except Exception as e:
            last_errs.append(f"{prov}:{e}")

    return False, "; ".join(last_errs[:5]) or "no free provider configured", None




# --- Initialize SQLite DB on startup ---
try:
    init_db()
    ensure_user_columns()
    ensure_auth_tables()  # make sure refresh_tokens exists
    ensure_schema_migrations()   # <--- add this call    
    ensure_wallet_tables()   # <-- add this if missing
    print("[hh] Database initialized ✅")
except Exception as e:
    print("[hh] Database init warning:", e)

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

# Use ONE CORS strategy. (A) Flask-Cors:
from flask_cors import CORS
CORS(app, resources={r"*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)





# ----------------------                                                                                                          
# Auth Blueprint
# ----------------------
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# unified responses
def ok(payload: dict | None = None, status: int = 200):
    return jsonify({"ok": True, **(payload or {})}), status

def err(message: str, status: int = 400, code: str = "bad_request", extra: dict | None = None):
    body = {"ok": False, "error": {"code": code, "message": message}}
    if extra: body["error"].update(extra)
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

# light background cleanup thread (non-blocking)
def _cleanup_loop():
    while True:
        try:
            # attempt database cleanup if available
            try:
                cleanup_old_logs()
            except Exception:
                pass
            time.sleep(6 * 60 * 60)
        except Exception as e:
            print("[hh] Cleanup error:", e)
            time.sleep(60)

threading.Thread(target=_cleanup_loop, daemon=True).start()

# --- Admin guard ---
def admin_required(fn):
    @wraps(fn)
    def _wrap(*a, **k):
        tok = (request.headers.get("X-Admin-Token") or "").strip()
        if not ADMIN_TOKEN:
            return jsonify({"error":"admin not configured"}), 500
        if tok != ADMIN_TOKEN:
            # tiny debug line you can keep or remove later
            app.logger.warning("admin_required: bad token len=%s", len(tok))
            return jsonify({"error":"admin auth failed"}), 401
        return fn(*a, **k)
    return _wrap


# ----------------------
# Simple utility endpoints                                                                                                       
# ----------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message":"Human Helper API is live", "status":"ok"}), 200

# Optional: don’t rate-limit health checks
@limiter.exempt
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"message":"Human Helper API is live", "status":"ok"}), 200

# ✅ Simple echo test route
@app.route("/echo", methods=["POST"])
def echo():
    return jsonify({"you_sent": request.json or {}})

@app.route("/version", methods=["GET"])
def version():
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("COMMIT_SHA", "dev")
    return jsonify({"commit": sha}), 200

@app.route("/debug/whichdb", methods=["GET"])
def debug_whichdb():
    import sqlite3
    db_file = os.environ.get("DATABASE_URL") or "db.sqlite3"
    if db_file.startswith("sqlite:///"):
        db_file = db_file.replace("sqlite:///", "")
    info = {"db_file": db_file, "exists": os.path.exists(db_file)}
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        info["tables"] = [r[0] for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        info["error"] = str(e)
    return jsonify(info)



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
            if ok1: return jsonify({"ok": True, "provider": "openai", "answer": res1}), 200

        # 2) Gemini
        ok2, res2 = gemini_free_chat(prompt, timeout=GEMINI_TIMEOUT)
        if ok2: return jsonify({"ok": True, "provider": "gemini", "answer": res2}), 200

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
        if ok4: return jsonify({"ok": True, "provider": "llama", "answer": res4}), 200

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
                model = tok.split(":",1)[1] if ":" in tok else "gpt-oss-20b"
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



# =======================                                                                                                       
# Auth blueprint + routes
# =======================

def _utcnow(): return datetime.now(timezone.utc)
def _otp(): return f"{random.randint(100000,999999)}"

MOBILE_IN_RE = re.compile(r'^[6-9]\d{9}$')          # India mobile
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
OTP_TTL_MIN = int(os.getenv("OTP_TTL_MIN","10"))
SEND_VERIFICATION_MODE = os.getenv("SEND_VERIFICATION_MODE","console")

def _send_code(dest: str, code: str):
    if SEND_VERIFICATION_MODE == "console":
        print(f"[VERIFY] send code {code} to {dest}")

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

def _clean(s): return (s or "").strip()

@auth_bp.post("/register")
@limiter.limit("10/hour")
def register():
    d = request.get_json(silent=True) or {}
    name     = _clean(d.get("full_name"))
    mobile   = _clean(d.get("mobile"))
    password = d.get("password") or ""
    email    = _clean(d.get("email"))
    address  = _clean(d.get("address"))

    if not name or len(name) < 2:              return jsonify({"error":"full_name required"}), 400
    if not MOBILE_IN_RE.match(mobile):         return jsonify({"error":"invalid mobile (India 10-digit)"}), 400
    if len(password) < 8:                      return jsonify({"error":"password must be at least 8 chars"}), 400
    if email and not EMAIL_RE.match(email):    return jsonify({"error":"invalid email"}), 400
    if len(address) < 4:                       return jsonify({"error":"address required"}), 400

    code    = _otp()

    expires = _now_ts() + OTP_TTL_MIN*60
    pwd_hash = pbkdf2_sha256.hash(password)

    db = get_db()
    cur = db.cursor()
    row = cur.execute("SELECT id,is_verified FROM users WHERE mobile=?", (mobile,)).fetchone()
    if row and int(row["is_verified"]) == 1:
        db.close()
        return jsonify({"error":"mobile already registered"}), 409

    # Upsert; keep user unverified until OTP verified
    cur.execute("""
        INSERT INTO users (name,mobile,email,address,password_hash,verification_code,verify_expires,is_verified)
        VALUES (?,?,?,?,?,?,?,0)
        ON CONFLICT(mobile) DO UPDATE SET
            name=excluded.name,
            email=excluded.email,
            address=excluded.address,
            password_hash=excluded.password_hash,
            verification_code=excluded.verification_code,
            verify_expires=excluded.verify_expires,
            is_verified=0
    """, (name, mobile, email, address, pwd_hash, code, expires))
    db.commit(); db.close()

    _send_code(mobile, code)
    return jsonify({"message":"verification code sent","mobile":mobile,"expires_in_min":OTP_TTL_MIN}), 201

@auth_bp.post("/verify")
@limiter.limit("20/hour")
def verify():
    d = request.get_json(silent=True) or {}
    mobile = _clean(d.get("mobile"))
    code   = _clean(d.get("code"))

    if not MOBILE_IN_RE.match(mobile):             return jsonify({"error":"invalid mobile"}), 400
    if not code.isdigit() or len(code) != 6:       return jsonify({"error":"invalid code"}), 400

    db = get_db()
    row = db.execute(
        "SELECT id,verification_code,verify_expires,is_verified FROM users WHERE mobile=?",
        (mobile,)
    ).fetchone()
    if not row:                       db.close(); return jsonify({"error":"user not found"}), 404
    if int(row["is_verified"]) == 1:  db.close(); return jsonify({"message":"already verified"}), 200
    if (row["verification_code"] or "") != code:
        db.close(); return jsonify({"error":"incorrect code"}), 400
    if row["verify_expires"] and _now_ts() > int(row["verify_expires"]):
        db.close(); return jsonify({"error":"code expired"}), 400

    db.execute("UPDATE users SET is_verified=1, verification_code=NULL, verify_expires=NULL WHERE id=?", (row["id"],))
    db.commit(); db.close()
    return jsonify({"message":"verified"}), 200

@auth_bp.post("/resend-code")
@limiter.limit("5/hour")
def resend_code():
    d = request.get_json(silent=True) or {}
    mobile = _clean(d.get("mobile"))
    if not MOBILE_IN_RE.match(mobile): return jsonify({"error":"invalid mobile"}), 400

    db = get_db()
    row = db.execute("SELECT id,is_verified FROM users WHERE mobile=?", (mobile,)).fetchone()
    if not row:                       db.close(); return jsonify({"error":"not registered"}), 404
    if int(row["is_verified"]) == 1:  db.close(); return jsonify({"message":"already verified"}), 200

    code = _otp()
    expires = _now_ts() + OTP_TTL_MIN*60
    db.execute("UPDATE users SET verification_code=?, verify_expires=? WHERE id=?", (code, expires, row["id"]))
    db.commit(); db.close()
    _send_code(mobile, code)
    return jsonify({"message":"verification code re-sent","expires_in_min":OTP_TTL_MIN}), 200

@auth_bp.post("/login")
@limiter.limit("5/minute;50/hour")
def login():
    d = request.get_json(silent=True) or {}
    mobile = _clean(d.get("mobile"))
    password = d.get("password") or ""
    if not MOBILE_IN_RE.match(mobile) or not password:
        return jsonify({"error":"invalid credentials"}), 400

    db = get_db()
    row = db.execute(
        "SELECT id,is_banned,is_verified,password_hash,name,email,address FROM users WHERE mobile=?",
        (mobile,)
    ).fetchone()
    db.close()

    if not row:                       return jsonify({"error":"user not found"}), 404
    if row["is_banned"]:              return jsonify({"error":"account banned"}), 403
    if not row["is_verified"]:        return jsonify({"error":"not verified"}), 403
    if not pbkdf2_sha256.verify(password, row["password_hash"]):
        return jsonify({"error":"invalid credentials"}), 401

    # ✅ these MUST be inside the function and aligned here
    access  = make_access_token(row["id"], mobile)
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
    row = conn.execute(
        "SELECT user_id, revoked, expires_at FROM refresh_tokens WHERE jti=?", (jti,)
    ).fetchone()
    if (not row) or row["revoked"] or row["expires_at"] < _now_ts():
        conn.close()
        return jsonify({"error": "refresh revoked/expired"}), 401

    # rotate: revoke current, issue new
    conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE jti=?", (jti,))
    conn.commit()   # ✅ must be indented to the same level as conn.execute

    # fetch mobile for claim
    mrow = conn.execute(
        "SELECT mobile FROM users WHERE id=?", (row["user_id"],)
    ).fetchone()
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
        return jsonify({"error":"invalid refresh"}), 400
    jti = payload.get("jti","")
    conn = get_db()
    conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE jti=?", (jti,))
    conn.commit()
    conn.close()
    return jsonify({"message":"logged out"}), 200

@app.before_request
def _maybe_purge():
    # ~1% of requests purge old refresh tokens
    if random.randint(1,100) == 1:
        try: clean_expired_refresh()
        except Exception: pass

@app.get("/whoami")
@auth_required
def whoami():
    return jsonify({"user_id": g.user_id, "mobile": g.mobile})

# register blueprint once
app.register_blueprint(auth_bp)



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
        s = cur.execute("SELECT balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        if not s: raise ValueError("sender not found")
        if float(s["balance"]) < amount: raise ValueError("insufficient balance")
        if not cur.execute("SELECT 1 FROM users WHERE id=?", (receiver_id,)).fetchone():
            raise ValueError("receiver not found")

        cur.execute("UPDATE users SET balance = balance - ? WHERE id=?", (amount, g.user_id))
        r1 = cur.execute("SELECT balance, locked_balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        cur.execute("""INSERT INTO wallet_txns
            (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
            VALUES (?,?,?,?,?,'success',NULL,?,NULL)
        """, (g.user_id, "p2p_out", -amount, r1["balance"], r1["locked_balance"], f"to user:{receiver_id}"))

        cur.execute("UPDATE users SET balance = balance + ? WHERE id=?", (amount, receiver_id))
        r2 = cur.execute("SELECT balance, locked_balance FROM users WHERE id=?", (receiver_id,)).fetchone()
        cur.execute("""INSERT INTO wallet_txns
            (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
            VALUES (?,?,?,?,?,'success',NULL,?,NULL)
        """, (receiver_id, "p2p_in", amount, r2["balance"], r2["locked_balance"], f"from user:{g.user_id}"))

        db.commit()
    except Exception as e:
        db.rollback(); db.close()
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
        r = cur.execute("SELECT balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        if not r or float(r["balance"]) < amount:
            raise ValueError("insufficient balance")

        # move to locked
        cur.execute("""
            UPDATE users
            SET balance = balance - ?, locked_balance = locked_balance + ?
            WHERE id=?""", (amount, amount, g.user_id))
        rlock = cur.execute("SELECT balance, locked_balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        cur.execute("""INSERT INTO wallet_txns
            (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
            VALUES (?,?,?,?,?,'requested',?, ?, ?)
        """, (g.user_id, "withdraw_lock", -amount, rlock["balance"], rlock["locked_balance"],
              None, f"lock for withdraw {amount}",
              json.dumps({"upi": upi_id, "net": net, "fee": fee_total})))

        # record withdrawal request as processing
        cur.execute("""INSERT INTO withdrawal_requests
            (user_id, amount, net_amount, fee_amount, upi, status)
            VALUES (?,?,?,?,?,'processing')""", (g.user_id, amount, net, fee_total, upi_id))
        wreq_id = cur.lastrowid
        db.commit()
    except Exception as e:
        db.rollback(); db.close()
        return err(str(e), 400)

    # call Razorpay outside transaction
    ok_pay, res = _razorpay_payout_upi(upi_id, net, ref=f"WREQ:{wreq_id}")

    db = get_db(); cur = db.cursor()
    try:
        db.execute("BEGIN IMMEDIATE")
        if not ok_pay:
            # release lock
            cur.execute("""
                UPDATE users
                SET balance = balance + ?, locked_balance = locked_balance - ?
                WHERE id=?""", (amount, amount, g.user_id))
            rrel = cur.execute("SELECT balance, locked_balance FROM users WHERE id=?", (g.user_id,)).fetchone()
            cur.execute("""INSERT INTO wallet_txns
                (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
                VALUES (?,?,?,?,?,'failed',?, ?, ?)""",
                (g.user_id, "withdraw_release", amount, rrel["balance"], rrel["locked_balance"],
                 f"WREQ:{wreq_id}", "payout failed", json.dumps({"err": str(res)[:240]})))
            cur.execute("UPDATE withdrawal_requests SET status='failed', reason=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        (str(res)[:240], wreq_id))
            db.commit(); db.close()
            return err("payout failed", 502, extra={"details": str(res)})

        # success: clear lock, credit pooled fee, mark paid
        payout_id = res.get("id") if isinstance(res, dict) else None
        cur.execute("UPDATE users SET locked_balance = locked_balance - ? WHERE id=?", (amount, g.user_id))
        rset = cur.execute("SELECT balance, locked_balance FROM users WHERE id=?", (g.user_id,)).fetchone()
        cur.execute("""INSERT INTO wallet_txns
            (user_id,kind,amount,balance_after,locked_after,status,ref,note,meta)
            VALUES (?,?,?,?,?,'success',?, ?, NULL)""",
            (g.user_id, "withdraw_settle", 0.0, rset["balance"], rset["locked_balance"],
             f"WREQ:{wreq_id}", f"payout success: {payout_id}"))

        # add the ENTIRE fee to the single pool
        cur.execute("""
            UPDATE fee_pool
            SET pool_balance = pool_balance + ?, updated_at = CURRENT_TIMESTAMP
            WHERE id=1
        """, (fee_total,))

        cur.execute("UPDATE withdrawal_requests SET status='paid', payout_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (payout_id, wreq_id))
        db.commit()
    except Exception as e:
        db.rollback(); db.close()
        return err("post-payout update failed", 500, extra={"details": str(e)})
    db.close()

    return ok({
        "message": "withdrawal processed",
        "request_id": wreq_id,
        "net_paid": net,
        "fee": fee_total,
        "payout_id": payout_id
    })

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
# Start background cleanup thread (runs every 6h)
threading.Thread(target=_cleanup_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("[hh] Starting HumanHelper backend on port", port)
    print("[hh] OPENROUTER configured:", bool(OPENROUTER_API_KEY and OPENROUTER_URL))
    print("[hh] OpenAI configured:", bool(OPENAI_API_KEY))
    print("[hh] DEEPSEEK configured:", bool(DEEPSEEK_API_KEY))
    print("[hh] FREE_MODELS_PRIORITY:", FREE_MODELS_PRIORITY)
    app.run(host="0.0.0.0", port=port, debug=True)
