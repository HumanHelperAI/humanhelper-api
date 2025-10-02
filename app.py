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
from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS

# load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------
# Defensive local modules (stubs if missing)
# ----------------------
try:
    from database import init_db, run_query, cleanup_old_logs
except Exception:
    def init_db(*a, **k): 
        print("[hh] database.init_db stub")
    def run_query(sql, params=(), fetch=False):
        # safe stub; return empty for fetch, else None
        if fetch:
            return []
        return None
    def cleanup_old_logs():
        return
    print("[hh] Warning: database module not found — using stubs")

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

# Make sure we have sane defaults (fallback list if env empty)
if not FREE_MODELS_PRIORITY:
    FREE_MODELS_PRIORITY = [
        "openrouter:gpt-oss-20b",
        "openrouter:gpt-oss-120b",
        "deepseek:deepseek-v3.1-free",
        "deepseek:r1-free",
        "nousresearch:deephermes-3-llama-3-8b-preview",
        "google:gemini-2.0-flash-exp"
    ]

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

ALLOWED_ORIGINS = {
    "https://humanhelperai.in",
    "https://www.humanhelperai.in",
    "https://humanhelperai.github.io",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, X-Admin-Token, X-Premium"
        resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

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

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"message":"Human Helper API is live", "status":"ok"}), 200

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

# ----------------------
# Other user endpoints (signup/login/balance/transactions etc.) -- defensive stubs
# ----------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json or {}
    if not data.get("name") or not data.get("mobile") or not data.get("aadhar") or not data.get("password"):
        return jsonify({"error":"name, mobile, aadhar and password required"}), 400
    try:
        ok, _, wait_log, eta, err = enqueue_write(
            "INSERT INTO users (name, mobile, aadhar, pan, password) VALUES (?,?,?,?,?)",
            (data.get("name"), data.get("mobile"), data.get("aadhar"), data.get("pan",""), data.get("password")),
            timeout=10.0
        )
        if not ok:
            return jsonify({"error": err or "Signup failed", "wait_log": wait_log, "eta_seconds": eta}), 400
        return jsonify({"message":"User registered ✅", "wait_log": wait_log, "eta_seconds": eta}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/login", methods=["POST"])
def login():
    d = request.json or {}
    rows = run_query("SELECT balance,is_banned FROM users WHERE mobile=? AND password=?", (d.get("mobile"), d.get("password")), fetch=True)
    if rows:
        if rows[0][1]:
            return jsonify({"error":"Account banned ❌"}), 403
        return jsonify({"message":"Login successful ✅","balance": rows[0][0]})
    return jsonify({"error":"Invalid credentials ❌"}), 401

# ... remaining endpoints as in your previous full app can be re-added similarly ...
# For brevity, we've included the main ones above and left pattern for you to extend.

# ----------------------
# GitHub helpers
# ----------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

def github_get_file(owner: str, repo: str, path: str, ref: str = "main"):
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not configured"
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.raw"}
    ok, resp = _do_request_with_retries("get", url, headers=headers, timeout=10)
    if not ok:
        return False, f"GitHub request failed: {resp}"
    return True, resp.text

def github_create_gist(files: dict, description: str = "", public: bool = False):
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not configured"
    url = "https://api.github.com/gists"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Content-Type": "application/json"}
    payload = {"files": {name: {"content": content} for name, content in files.items()}, "description": description, "public": public}
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=10)
    if not ok:
        return False, f"GitHub gist creation failed: {resp}"
    try:
        j = resp.json()
        return True, j.get("html_url") or j.get("url")
    except Exception:
        return False, "GitHub gist parse error"

@app.route("/github/get", methods=["GET"])
def github_get():
    owner = request.args.get("owner")
    repo = request.args.get("repo")
    path = request.args.get("path")
    ref = request.args.get("ref", "main")
    if not (owner and repo and path):
        return jsonify({"error":"owner,repo,path required"}), 400
    ok, res = github_get_file(owner, repo, path, ref=ref)
    if not ok:
        return jsonify({"error": res}), 400
    return jsonify({"ok": True, "content": res})

@app.route("/github/gist", methods=["POST"])
@admin_required
def github_gist():
    d = request.json or {}
    files = d.get("files")
    desc = d.get("description", "")
    public = bool(d.get("public", False))
    if not files or not isinstance(files, dict):
        return jsonify({"error":"files dict required"}), 400
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
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("[hh] Starting HumanHelper backend on port", port)
    print("[hh] OPENROUTER configured:", bool(OPENROUTER_API_KEY and OPENROUTER_URL))
    print("[hh] OpenAI configured:", bool(OPENAI_API_KEY))
    print("[hh] DEEPSEEK configured:", bool(DEEPSEEK_API_KEY))
    print("[hh] FREE_MODELS_PRIORITY:", FREE_MODELS_PRIORITY)
    app.run(host="0.0.0.0", port=port, debug=True)
