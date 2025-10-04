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

try:
    from database import init_db, run_query, cleanup_old_logs
except Exception:
    print("[hh] Warning: database module not found — using SQLite helpers")

    def init_db():
        # create users table if missing
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

    def cleanup_old_logs():
        # nothing to do for SQLite demo
        pass

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

# OpenRouter wrapper (OpenRouter uses a chat-completions style)
def ask_openrouter(model: str, prompt: str, timeout: int = OPENROUTER_TIMEOUT) -> Tuple[bool, str]:
    if not OPENROUTER_API_KEY or not OPENROUTER_URL:
        return False, "openrouter not configured"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3
    }
    ok, resp = _do_request_with_retries("post", OPENROUTER_URL, headers=headers, json_body=payload, timeout=timeout)
    if not ok:
        if isinstance(resp, requests.Response):
            return False, f"OpenRouter error {resp.status_code}: {resp.text[:800]}"
        return False, str(resp)
    try:
        j = resp.json()
        # OpenRouter may return choices similar to OpenAI
        if isinstance(j, dict):
            if "choices" in j and j["choices"]:
                first = j["choices"][0]
                # message style
                if isinstance(first, dict):
                    txt = first.get("message", {}).get("content") or first.get("text") or first.get("output") or ""
                    return True, (txt or "").strip()
            # sometimes the provider returns direct fields
            for k in ("text", "output", "content", "response"):
                if k in j and isinstance(j[k], str):
                    return True, j[k].strip()
        return True, (resp.text or "")[:2000]
    except Exception as e:
        return False, f"openrouter parse error: {e}"

# OpenAI REST (generic)
def ask_openai_rest(prompt: str, timeout: int = OPENAI_TIMEOUT) -> Tuple[bool, str]:
    key = OPENAI_API_KEY
    if not key:
        return False, "openai not configured"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "messages": [{"role":"user","content": prompt}],
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS","400")),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE","0.3"))
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
        # fallback shapes
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
def gemini_free_chat(prompt: str, timeout: int = GEMINI_TIMEOUT) -> Tuple[bool,str]:
    if GEMINI_REST_URL and GEMINI_API_KEY:
        headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS","400"))}
        ok, resp = _do_request_with_retries("post", GEMINI_REST_URL, headers=headers, json_body=payload, timeout=timeout)
        if not ok:
            if isinstance(resp, requests.Response):
                return False, f"Gemini-free error {resp.status_code}: {resp.text[:800]}"
            return False, str(resp)
        try:
            j = resp.json()
            for k in ("content","text","answer","response"):
                if k in j and isinstance(j[k], str):
                    return True, j[k].strip()
            if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                first = j["choices"][0]
                for k in ("text","content","message"):
                    if k in first and isinstance(first[k], str):
                        return True, first[k].strip()
            return True, json.dumps(j)[:1000]
        except Exception as e:
            return False, f"Gemini-free parse error: {e}"
    return False, "Gemini-free not configured"

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

# ----------------------
# Model selection helper: try free providers in the configured order
# FREE_MODELS_PRIORITY format: "<provider>:<model>" e.g. "openrouter:gpt-oss-20b"
# Provider keys we support: openrouter, deepseek, openai, gemini, llama (llama handled separately)
# ----------------------
def try_free_models_in_order(prompt: str, timeout_each: int = OPENROUTER_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """
    Returns (ok, answer, provider_name) where provider_name is descriptive.
    """
    last_errs = []
    for token in FREE_MODELS_PRIORITY:
        if ":" in token:
            prov, model = token.split(":",1)
            prov = prov.strip().lower()
            model = model.strip()
        else:
            # if only provider given, choose a default model name
            prov = token.strip().lower()
            model = ""
        try:
            if prov in ("openrouter", "openrouter.ai", "openrouter_api"):
                if not OPENROUTER_API_KEY:
                    last_errs.append("openrouter not configured")
                    continue
                ok, res = ask_openrouter(model or "gpt-oss-20b", prompt, timeout=timeout_each)
                if ok: return True, res, f"openrouter/{model or 'default'}"
                last_errs.append(f"openrouter:{res}")
            elif prov in ("deepseek", "deepseek.com"):
                if not DEEPSEEK_API_KEY:
                    last_errs.append("deepseek not configured")
                    continue
                ok, res = ask_deepseek(prompt, timeout=timeout_each)
                if ok: return True, res, "deepseek"
                last_errs.append(f"deepseek:{res}")
            elif prov in ("openai", "openai.com", "gpt"):
                if not OPENAI_API_KEY:
                    last_errs.append("openai not configured")
                    continue
                ok, res = ask_openai_rest(prompt, timeout=timeout_each)
                if ok: return True, res, "openai"
                last_errs.append(f"openai:{res}")
            elif prov.startswith("google") or prov.startswith("gemini"):
                ok, res = gemini_free_chat(prompt, timeout=timeout_each)
                if ok: return True, res, "gemini-free"
                last_errs.append(f"gemini:{res}")
            elif prov in ("nous", "nousresearch", "nousresearch/deephermes"):
                # try via openrouter or deepseek if configured - handled above if token explicitly specified
                last_errs.append("nous: not directly supported; try via openrouter if available")
            else:
                last_errs.append(f"{prov}:unknown provider")
        except Exception as e:
            last_errs.append(str(e))
    # none responded
    return False, "; ".join(last_errs[:5]) or "no free provider configured", None

# ----------------------
# Admin blueprint + helpers
# ----------------------
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = request.headers.get("X-Admin-Token", "") or ""
        if token != ADMIN_TOKEN and token != os.getenv("ADMIN_TOKEN", ADMIN_TOKEN):
            return jsonify({"error":"admin auth failed"}), 403
        return fn(*args, **kwargs)
    return wrapper

def _audit(action, actor, target, details):
    ip = request.headers.get("X-Forwarded-For") or request.remote_addr or "-"
    try:
        enqueue_write("INSERT INTO audit_log (action, actor, target, details, ip) VALUES (?,?,?,?,?)",
                      (action, actor, target, details, ip), timeout=3.0)
    except Exception:
        pass

@admin_bp.get("/users")
@admin_required
def list_users():
    q = (request.args.get("q") or "").strip()
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    params, sql = [], "SELECT id,name,mobile,aadhar,pan,balance,is_banned FROM users"
    if q:
        sql += " WHERE name LIKE ? OR mobile LIKE ? OR pan LIKE ?"
        like = f"%{q}%"
        params.extend([like, like, like])
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = run_query(sql, tuple(params), fetch=True)
    users = [{"id": r[0], "name": r[1], "mobile": r[2], "aadhar": r[3],
              "pan": r[4], "balance": r[5], "is_banned": bool(r[6])} for r in rows]
    return jsonify({"users": users})

@admin_bp.post("/user/ban")
@admin_required
def user_ban():
    data = request.json or {}
    mobile, reason = data.get("mobile"), data.get("reason","")
    if not mobile:
        return jsonify({"error":"mobile required"}), 400
    ok, _, log, eta, err = enqueue_write("UPDATE users SET is_banned=1 WHERE mobile=?", (mobile,), timeout=10.0)
    _audit("user_ban", "admin", mobile, reason)
    if not ok:
        return jsonify({"error": err or "ban failed", "wait_log": log, "eta": eta}), 400
    return jsonify({"message": f"user {mobile} banned ✅", "wait_log": log, "eta": eta}), 200

# ... (other admin routes can be added similarly) ...

# ----------------------
# Flask app init
# ----------------------
app = Flask(__name__)

# --- Initialize SQLite DB on startup ---
try:
    init_db()
    ensure_user_columns()
    ensure_auth_tables()  # make sure refresh_tokens exists
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

# If you prefer manual headers instead, keep add_cors_headers() below and delete the CORS(...) line.

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])
# Example per-route: @limiter.limit("10/minute")

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

# register admin blueprint
try:
    app.register_blueprint(admin_bp)
except Exception as e:
    print("[hh] admin blueprint register failed:", e)

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
def is_premium_user(req) -> bool:
    try:
        j = req.get_json(silent=True) or {}
        if isinstance(j.get("premium"), bool):
            return j.get("premium")
    except Exception:
        pass
    h = (req.headers.get("X-Premium") or "").strip().lower()
    if h in ("1", "true", "yes", "on"):
        return True
    admin_token = req.headers.get("X-Admin-Token", "")
    if admin_token and admin_token == ADMIN_TOKEN:
        return True
    return False

@app.route("/ai/ask", methods=["POST"])
def ai_ask():
    """
    Lightweight single-call AI endpoint (tries free-first configured providers).
    This is best-effort and will return first working provider's answer.
    """
    data = request.get_json(force=False) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error":"prompt required"}), 400

    # Try free models in order (OpenRouter / DeepSeek / OpenAI if listed)
    ok, ans_or_err, provider = try_free_models_in_order(prompt, timeout_each=10)
    if ok:
        return jsonify({"ok": True, "provider": provider or "free", "answer": ans_or_err}), 200

    # Fallback: if premium and OpenAI configured, try OpenAI
    if is_premium_user(request) and OPENAI_API_KEY:
        ok2, res2 = ask_openai_rest(prompt, timeout=OPENAI_TIMEOUT)
        if ok2:
            return jsonify({"ok": True, "provider":"openai", "answer": res2}), 200
        else:
            return jsonify({"ok": False, "provider":"openai", "error": res2}), 502

    # Try DeepSeek as a last attempt
    if DEEPSEEK_API_KEY:
        ok3, res3 = ask_deepseek(prompt, timeout=DEEPSEEK_TIMEOUT)
        if ok3:
            return jsonify({"ok": True, "provider":"deepseek", "answer": res3}), 200
        else:
            return jsonify({"ok": False, "provider":"deepseek", "error": res3}), 502

    return jsonify({"ok": False, "provider":"none", "answer": ans_or_err}), 502

@app.route("/ai/humanhelper", methods=["POST"])
def ai_humanhelper():
    """
    Rich routing: non-premium tries free models only (no paid OpenAI),
    premium tries paid providers first (OpenAI -> DeepSeek -> LLaMA).
    """
    data = request.get_json(force=False) or {}
    prompt = (data.get("prompt") or "").strip()
    history = data.get("history", None)
    if not prompt:
        return jsonify({"error":"prompt required"}), 400

    premium = is_premium_user(request)

    # Non-premium: try free models first (OpenRouter/DeepSeek)
    if not premium:
        ok, ans, prov = try_free_models_in_order(prompt, timeout_each=10)
        if ok:
            return jsonify({"provider": prov or "free", "ok": True, "answer": ans}), 200
        # LLaMA fallback (local) if configured
        llama_ok, llama_ans = ask_llama(prompt, timeout=LLAMA_TIMEOUT)
        if llama_ok:
            return jsonify({"provider":"llama","ok":True,"answer":llama_ans}), 200
        return jsonify({"provider":"none","ok":False,"answer": ans}), 502

    # Premium path:
    # 1) OpenAI if configured
    if OPENAI_API_KEY:
        ok_oa, ans_oa = ask_openai_rest(prompt, timeout=OPENAI_TIMEOUT)
        if ok_oa:
            return jsonify({"provider":"openai","ok":True,"answer":ans_oa}), 200

    # 2) DeepSeek if configured
    if DEEPSEEK_API_KEY:
        ok_ds, ans_ds = ask_deepseek(prompt, timeout=DEEPSEEK_TIMEOUT)
        if ok_ds:
            return jsonify({"provider":"deepseek","ok":True,"answer":ans_ds}), 200

    # 3) OpenRouter as paid fallback if configured
    for tok in FREE_MODELS_PRIORITY:
        if tok.startswith("openrouter"):
            # attempt with default model if not specified
            model = tok.split(":",1)[1] if ":" in tok else "gpt-oss-20b"
            ok_or, res_or = ask_openrouter(model, prompt, timeout=OPENROUTER_TIMEOUT)
            if ok_or:
                return jsonify({"provider": f"openrouter/{model}", "ok": True, "answer": res_or}), 200

    # 4) LLaMA fallback
    llama_ok, llama_ans = ask_llama(prompt, timeout=LLAMA_TIMEOUT)
    if llama_ok:
        return jsonify({"provider":"llama","ok":True,"answer":llama_ans}), 200

    return jsonify({"provider":"none","ok":False,"answer":"All providers failed. Please try again later."}), 502

@app.route("/ai/status", methods=["GET"])
def ai_status():
    return jsonify({
        "openai_configured": bool(OPENAI_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "gemini_configured": bool(GEMINI_REST_URL and GEMINI_API_KEY),
        "llama_server": LLAMA_SERVER_URL,
        "free_models_priority": FREE_MODELS_PRIORITY,
        "notes": {
            "free_first": "Tries free providers in FREE_MODELS_PRIORITY order (OpenRouter/DeepSeek/OpenAI/Gemini...).",
            "premium_flow": "OpenAI -> DeepSeek -> OpenRouter -> LLaMA fallback."
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
