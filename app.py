#!/usr/bin/env python3
# app.py
"""
HumanHelper backend (final consolidated version)

Behavior summary:
- Non-premium: Gemini-free (REST via GEMINI_REST_URL+GEMINI_API_KEY or SDK) -> LLaMA fallback
  (do NOT call OpenAI/DeepSeek for non-premium to save money)
- Premium: OpenAI -> DeepSeek -> LLaMA fallback
- LLaMA server URL is env LLAMA_SERVER_URL (default http://127.0.0.1:5001/completion)
- Admin auth via X-Admin-Token header matching ADMIN_TOKEN env (default "changeme")
- Defensive stubs for local modules if missing: database, writer, wallet, earnings
- GitHub helper: GET raw file + create gist (requires GITHUB_TOKEN env)
"""

from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from functools import wraps
import os
import time
import threading
import random
import requests
import json
from typing import Tuple

# Load environment from .env if present
from dotenv import load_dotenv
load_dotenv()

# ------------------------ Local project modules (defensive) ------------------------
# Your project should supply these modules; if missing we use harmless stubs so app stays up.
try:
    from database import init_db, run_query, cleanup_old_logs
except Exception:
    def init_db(*a, **k): pass
    def run_query(sql, params=(), fetch=False):
        # mimic fetch returning empty list
        if fetch:
            return []
        return None
    def cleanup_old_logs():
        return
    print("Warning: database module not found — using stubs")

try:
    from writer import start_writer, enqueue_write
except Exception:
    # enqueue_write stub returns (ok, result, wait_log, eta, err)
    def start_writer(): pass
    def enqueue_write(sql, params=(), timeout=1.0):
        # Return queued-like response for compatibility
        return True, None, ["immediate"], 0, None
    print("Warning: writer module not found — using stub enqueue_write")

try:
    from wallet import deposit, withdraw
except Exception:
    def deposit(mobile, amount):
        return False, "wallet not configured"
    def withdraw(mobile, amount):
        return False, "wallet not configured"
    print("Warning: wallet module not found — using stubs")

try:
    from earnings import reward_user
except Exception:
    def reward_user(mobile, video_id, content_type, duration):
        return False, "earnings not configured"
    print("Warning: earnings module not found — using stub reward_user")

# ------------------------ Config / env ------------------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_REST_URL = os.getenv("GEMINI_REST_URL", "")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "20"))
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:5001/completion")
LLAMA_N_PREDICT = int(os.getenv("LLAMA_N_PREDICT", "256"))
LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", "30"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "20"))
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "20"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# If you maintain a list of free models to try (openrouter), set as comma separated env FREE_MODELS_PRIORITY
FREE_MODELS_PRIORITY = [m.strip() for m in os.getenv("FREE_MODELS_PRIORITY", "").split(",") if m.strip()]

# Try to import Gemini SDK (if you prefer SDK usage)
GEMINI_SDK_OK = False
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_SDK_OK = True
except Exception:
    genai = None
    GEMINI_SDK_OK = False

# ------------------------ Utilities ------------------------
def _do_request_with_retries(method, url, headers=None, json_body=None, timeout=15, max_attempts=3):
    attempt = 0
    backoff_base = 0.6
    last_exc = None
    while attempt < max_attempts:
        try:
            if method.lower() == "post":
                r = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            else:
                r = requests.get(url, headers=headers, params=json_body, timeout=timeout)
            if 200 <= r.status_code < 300:
                return True, r
            # handle rate-limit / service-unavailable
            if r.status_code in (429, 503) and 'Retry-After' in r.headers:
                try:
                    ra = int(r.headers.get('Retry-After', '0'))
                except Exception:
                    ra = 0
                wait = ra + random.uniform(0.2, 0.6)
                time.sleep(wait)
                attempt += 1
                continue
            # transient fallback with jitter
            if r.status_code in (429, 503):
                time.sleep(backoff_base * (2 ** attempt) + random.uniform(0, 0.3))
                attempt += 1
                continue
            return False, r
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(backoff_base * (2 ** attempt) + random.uniform(0, 0.3))
            attempt += 1
    return False, f"request failed after {max_attempts} attempts: {last_exc}"

# ------------------------ OpenAI wrapper (REST) ------------------------
def ask_openai_rest(prompt: str, timeout: int = OPENAI_TIMEOUT) -> Tuple[bool, str]:
    key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    if not key:
        return False, "OpenAI not configured"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS","400")),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE","0.3"))
    }
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=body, timeout=timeout)
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
        return True, json.dumps(j)[:1000]
    except Exception as e:
        return False, f"OpenAI parse error: {e}"

# ------------------------ DeepSeek wrapper ------------------------
def ask_deepseek(prompt: str, timeout: int = DEEPSEEK_TIMEOUT) -> Tuple[bool, str]:
    key = os.getenv("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    if not key:
        return False, "DeepSeek not configured"
    url = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "400")),
        "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.6"))
    }
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=timeout)
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
        return True, json.dumps(j)[:1000]
    except Exception as e:
        return False, f"DeepSeek parse error: {e}"

# ------------------------ Gemini-free wrapper (REST or SDK) ------------------------
def gemini_free_chat(prompt: str, timeout: int = GEMINI_TIMEOUT) -> Tuple[bool,str]:
    # 1) Prefer REST proxy when GEMINI_REST_URL and GEMINI_API_KEY are configured (this is your "free" remote)
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
            # common shapes
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
    # 2) If SDK is installed and configured, try it
    if GEMINI_SDK_OK and genai:
        try:
            # using simple generation; exact usage may depend on SDK version
            model = genai.GenerativeModel("gemini")
            resp = model.generate_content(prompt)
            return True, (resp.text or "").strip()
        except Exception as e:
            return False, f"Gemini SDK error: {e}"
    return False, "Gemini-free not configured"

# ------------------------ LLaMA local server wrapper ------------------------
def _parse_llama_response_obj(j):
    if not isinstance(j, dict):
        return None
    for k in ("content","text","result","answer","output","response"):
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    choices = j.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            for k in ("text","content","message"):
                v = first.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return None

def ask_llama(prompt: str, n_predict: int = LLAMA_N_PREDICT, timeout: int = LLAMA_TIMEOUT) -> Tuple[bool,str]:
    url = os.getenv("LLAMA_SERVER_URL", LLAMA_SERVER_URL)
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
        text = _parse_llama_response_obj(j)
        if text:
            return True, text
        return True, (r.text or "").strip()
    except Exception:
        return True, (r.text or "").strip() or ""

def ask_humanhelper_llama_chat(prompt: str, history: list | None = None,
                               n_predict: int = LLAMA_N_PREDICT, timeout: int = LLAMA_TIMEOUT) -> Tuple[bool,str]:
    system_prompt = "You are HumanHelper.ai — helpful assistant for local services and recommendations."
    parts = [f"<|system|>\n{system_prompt}\n"]
    if history:
        for m in history:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content: continue
            if role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
            else:
                parts.append(f"<|user|>\n{content}\n")
    parts.append(f"<|user|>\n{prompt}\n")
    assembled = "\n".join(parts)
    return ask_llama(assembled, n_predict=int(n_predict), timeout=timeout)

# ------------------------ GitHub helpers ------------------------
def github_get_file(owner: str, repo: str, path: str, ref: str = "main"):
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not configured"
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.raw"}
    ok, resp = _do_request_with_retries("get", url, headers=headers, timeout=15)
    if not ok:
        return False, f"GitHub request failed: {resp}"
    return True, resp.text

def github_create_gist(files: dict, description: str = "", public: bool = False):
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not configured"
    url = "https://api.github.com/gists"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Content-Type": "application/json"}
    payload = {"files": {name: {"content": content} for name, content in files.items()}, "description": description, "public": public}
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=15)
    if not ok:
        return False, f"GitHub gist creation failed: {resp}"
    try:
        j = resp.json()
        return True, j.get("html_url") or j.get("url")
    except Exception:
        return False, "GitHub gist parse error"

# ------------------------ Admin blueprint ------------------------
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

@admin_bp.post("/user/unban")
@admin_required
def user_unban():
    data = request.json or {}
    mobile = data.get("mobile")
    if not mobile:
        return jsonify({"error":"mobile required"}), 400
    ok, _, log, eta, err = enqueue_write("UPDATE users SET is_banned=0 WHERE mobile=?", (mobile,), timeout=10.0)
    _audit("user_unban", "admin", mobile, "")
    if not ok:
        return jsonify({"error": err or "unban failed", "wait_log": log, "eta": eta}), 400
    return jsonify({"message": f"user {mobile} unbanned ✅", "wait_log": log, "eta": eta}), 200

@admin_bp.post("/user/delete")
@admin_required
def user_delete():
    data = request.json or {}
    mobile, pan = data.get("mobile"), data.get("pan")
    if not (mobile or pan):
        return jsonify({"error":"mobile or pan required"}), 400
    field, val = ("mobile", mobile) if mobile else ("pan", pan)
    ok, _, log, eta, err = enqueue_write(f"DELETE FROM users WHERE {field}=?", (val,), timeout=10.0)
    _audit("user_delete", "admin", val, "")
    if not ok:
        return jsonify({"error": err or "delete failed", "wait_log": log, "eta": eta}), 400
    return jsonify({"message": f"user {val} deleted ✅", "wait_log": log, "eta": eta}), 200

@admin_bp.post("/balance/adjust")
@admin_required
def balance_adjust():
    data = request.json or {}
    mobile = data.get("mobile")
    try:
        delta = float(data.get("delta", 0))
    except Exception:
        return jsonify({"error":"invalid delta"}), 400
    note = data.get("note", "")
    if not mobile or delta == 0:
        return jsonify({"error":"mobile and non-zero delta required"}), 400
    ok1, _, l1, e1, err1 = enqueue_write("UPDATE users SET balance=balance+? WHERE mobile=?", (delta, mobile), timeout=10.0)
    ok2, _, l2, e2, err2 = enqueue_write("INSERT INTO transactions (mobile,type,amount,status,admin_note) VALUES (?,?,?,?,?)",
                                        (mobile, "admin_adjust", abs(delta), "completed", note), timeout=10.0)
    _audit("balance_adjust", "admin", mobile, f"delta={delta}; note={note}")
    if ok1 and ok2:
        return jsonify({"message": f"balance adjusted by ₹{delta} ✅", "wait_log": l1 + l2, "eta": max(e1, e2)}), 200
    return jsonify({"error": err1 or err2 or "adjust failed", "wait_log": l1 + l2, "eta": max(e1, e2)}), 400

@admin_bp.get("/transactions")
@admin_required
def tx_list():
    mobile = request.args.get("mobile")
    status = request.args.get("status")
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    sql = "SELECT id,mobile,type,amount,status,admin_note,timestamp FROM transactions"
    params = []
    clauses = []
    if mobile:
        clauses.append("mobile=?"); params.append(mobile)
    if status:
        clauses.append("status=?"); params.append(status)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"; params.extend([limit, offset])
    rows = run_query(sql, tuple(params), fetch=True)
    data = [{"id": r[0], "mobile": r[1], "type": r[2], "amount": r[3], "status": r[4], "admin_note": r[5], "timestamp": r[6]} for r in rows]
    return jsonify({"transactions": data})

@admin_bp.get("/audit")
@admin_required
def audit_list():
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    rows = run_query("SELECT id,action,actor,target,details,ip,timestamp FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset), fetch=True)
    data = [{"id": r[0], "action": r[1], "actor": r[2], "target": r[3], "details": r[4], "ip": r[5], "timestamp": r[6]} for r in rows]
    return jsonify({"audit": data})

# ------------------------ App init ------------------------
#!/usr/bin/env python3
# app.py - minimal health + CORS example for Railway / local dev
import time
import threading

app = Flask(__name__)

# IMPORTANT: allow only your GitHub Pages origin (no wildcard)
# Replace with your exact Github Pages origin if different.
ALLOWED_ORIGINS = ["https://humanhelperai.github.io"]

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# (Optional) If you want to enable credentials later, set supports_credentials=True
# CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)

# ------------------------ Simple endpoints ------------------------
@app.route("/", methods=["GET"])
def root():
    # simple root for quick checks
    return jsonify({"message": "Human Helper API is live", "status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    # lightweight health endpoint
    return jsonify({"message": "Human Helper API is live", "status": "ok"}), 200

# ------------------------ background cleanup (no-op safe example) ------------------------
def _cleanup_loop():
    while True:
        try:
            # if you have cleanup_old_logs() keep it; here we sleep harmlessly
            time.sleep(6 * 60 * 60)
        except Exception as e:
            print("Cleanup error:", e)
            time.sleep(60)

# start background thread (daemon so it won't block shutdown)
threading.Thread(target=_cleanup_loop, daemon=True).start()

@app.after_request
def add_cors_headers(response):
    # Avoid duplicating or misspelling header names. Only add extra helpful headers.
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# NOTE: These are expected to be defined elsewhere in your project:
# init_db(), start_writer(), admin_bp, cleanup_old_logs(), run_query()
# If they're missing you will see warning messages (we keep these safe so app still runs).

try:
    init_db()
except NameError:
    print("Warning: init_db() not defined in this module (expected elsewhere).")

try:
    start_writer()
except NameError:
    print("Warning: start_writer() not defined in this module (expected elsewhere).")

try:
    app.register_blueprint(admin_bp)
except NameError:
    print("Warning: admin_bp not defined in this module (expected elsewhere).")

# ------------------------ Run (for local dev only) ------------------------
if __name__ == "__main__":
    # For local dev, run on 127.0.0.1:5000
    # On Railway the gunicorn/Procfile will run the app normally.
    app.run(host="127.0.0.1", port=5000, debug=True)

# ------------------------ Wallet helpers normalization ------------------------
def _normalize_wallet_result(res):
    if not isinstance(res, (list, tuple)):
        return False, "invalid response", [], 0
    if len(res) == 2:
        ok, msg = res
        return ok, msg, [], 0
    if len(res) >= 4:
        ok, msg, wait_log, eta = res[0], res[1], res[2], res[3]
        return ok, msg, wait_log or [], eta or 0
    return False, "invalid response shape", [], 0

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

@app.route("/balance/<mobile>", methods=["GET"])
def balance(mobile):
    r = run_query("SELECT balance FROM users WHERE mobile=?", (mobile,), fetch=True)
    return jsonify({"mobile": mobile, "balance": r[0][0] if r else 0.0})

@app.route("/deposit", methods=["POST"])
def deposit_route():
    d = request.json or {}
    try:
        res = deposit(d["mobile"], float(d["amount"]))
        ok, msg, wait_log, eta = _normalize_wallet_result(res)
        return jsonify({"mobile": d["mobile"], "message": msg, "wait_log": wait_log, "eta_seconds": eta}), (200 if ok else 400)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/withdraw", methods=["POST"])
def withdraw_route():
    d = request.json or {}
    try:
        res = withdraw(d["mobile"], float(d["amount"]))
        ok, msg, wait_log, eta = _normalize_wallet_result(res)
        return jsonify({"mobile": d["mobile"], "message": msg, "wait_log": wait_log, "eta_seconds": eta}), (200 if ok else 400)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/transactions/<mobile>", methods=["GET"])
def transactions(mobile):
    txns = run_query("SELECT id,type,amount,status,admin_note,timestamp FROM transactions WHERE mobile=? ORDER BY timestamp DESC", (mobile,), fetch=True)
    return jsonify({"mobile": mobile, "transactions": [{"id": i, "type": t, "amount": a, "status": s, "admin_note": n, "timestamp": ts} for i, t, a, s, n, ts in txns]})

@app.route("/earn", methods=["POST"])
def earn():
    d = request.json or {}
    ok, res = reward_user(d.get("mobile"), d.get("video_id"), d.get("content_type"), int(d.get("duration", 0)))
    if not ok:
        return jsonify({"message": res}), 400
    new_bal = run_query("SELECT balance FROM users WHERE mobile=?", (d.get("mobile"),), fetch=True)[0][0]
    return jsonify({"mobile": d.get("mobile"), "message": f"Earned ₹{res['earned']} ✅ (₹{res['charity']} → charity)", "new_balance": new_bal})

@app.route("/charity/balance", methods=["GET"])
def charity_balance():
    r = run_query("SELECT balance FROM charity_wallet WHERE id=1", fetch=True)
    return jsonify({"charity_balance": r[0][0] if r else 0.0})

# ------------------------ Pilot: orders/rides/cherry & user endpoints ------------------------
# Insert after /charity/balance or similar user endpoints

@app.route("/user/transactions/<mobile>", methods=["GET"])
def user_transactions(mobile):
    """
    Public endpoint returning last 200 transactions for a mobile (read-only).
    """
    try:
        rows = run_query("SELECT id,type,amount,status,admin_note,timestamp FROM transactions WHERE mobile=? ORDER BY id DESC LIMIT 200", (mobile,), fetch=True)
        txs = [{"id": r[0], "type": r[1], "amount": r[2], "status": r[3], "admin_note": r[4], "timestamp": r[5]} for r in rows]
        return jsonify({"mobile": mobile, "transactions": txs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rewards/list/<mobile>", methods=["GET"])
def rewards_list_mobile(mobile):
    """
    List last N rewards for the mobile.
    """
    try:
        limit = int(request.args.get("limit", 50))
        rows = run_query("SELECT id,source,amount,charity_amount,timestamp FROM rewards WHERE mobile=? ORDER BY id DESC LIMIT ?", (mobile, limit), fetch=True)
        out = [{"id": r[0], "source": r[1], "amount": r[2], "charity": r[3], "timestamp": r[4]} for r in rows]
        return jsonify({"mobile": mobile, "rewards": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/orders/create", methods=["POST"])

@app.route("/orders/create", methods=["POST"])
def create_order_public():
    """
    Create a grocery/food order. Payload expected in JSON body:
    { "mobile": "...", "type": "grocery"|"food", "payload": { ... } }
    We'll enqueue DB writes using enqueue_write for reliability, with a direct fallback.
    """
    d = request.json or {}
    mobile = d.get("mobile")
    otype = d.get("type", "grocery")
    payload = json.dumps(d.get("payload", {}))
    if not mobile:
        return jsonify({"error":"mobile required"}), 400
    try:
        ok, _, wait_log, eta, err = enqueue_write(
            "INSERT INTO orders (mobile,type,payload,status) VALUES (?,?,?,?)",
            (mobile, otype, payload, "new"), timeout=5.0
        )
        if not ok:
            try:
                run_query("INSERT INTO orders (mobile,type,payload,status) VALUES (?,?,?,?)",
                          (mobile, otype, payload, "new"))
                _audit("order_create_fallback", mobile, otype, payload)
            except Exception as e2:
                return jsonify({"error": err or "order_create_failed", "wait_log": wait_log, "eta": eta, "fallback_error": str(e2)}), 500
        _audit("order_create", mobile, otype, payload)
        return jsonify({"message":"order_created", "wait_log": wait_log, "eta": eta}), 201
    except Exception as e:
        try:
            run_query("INSERT INTO orders (mobile,type,payload,status) VALUES (?,?,?,?)",
                      (mobile, otype, payload, "new"))
            _audit("order_create_exception_fallback", mobile, otype, payload)
            return jsonify({"message":"order_created_exception_fallback"}), 201
        except Exception as e2:
            return jsonify({"error": "order_create_failed_both", "detail": f"{e} | fallback: {e2}"}), 500


@app.route("/orders/list/<mobile>", methods=["GET"])
def orders_list_mobile(mobile):
    try:
        rows = run_query("SELECT id,type,payload,status,timestamp FROM orders WHERE mobile=? ORDER BY id DESC LIMIT 200", (mobile,), fetch=True)
        orders = [{"id": r[0], "type": r[1], "payload": r[2], "status": r[3], "timestamp": r[4]} for r in rows]
        return jsonify({"mobile": mobile, "orders": orders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rides/request", methods=["POST"])

@app.route("/rides/request", methods=["POST"])
def rides_request_public():
    """
    Simple ride request: body { "mobile": "...", "pickup": "...", "dropoff": "..." }
    We'll estimate fare with a simple heuristic and insert into rides table.
    """
    d = request.json or {}
    mobile = d.get("mobile")
    pickup = d.get("pickup")
    dropoff = d.get("dropoff")
    if not (mobile and pickup and dropoff):
        return jsonify({"error":"mobile, pickup, dropoff required"}), 400
    try:
        fare_est = float(os.getenv("RIDE_BASE_FARE", 40.0))
        ok, _, wait_log, eta, err = enqueue_write(
            "INSERT INTO rides (mobile,pickup,dropoff,status,fare_estimate) VALUES (?,?,?,?,?)",
            (mobile, pickup, dropoff, "requested", fare_est), timeout=5.0
        )
        if not ok:
            try:
                run_query("INSERT INTO rides (mobile,pickup,dropoff,status,fare_estimate) VALUES (?,?,?,?,?)",
                          (mobile, pickup, dropoff, "requested", fare_est))
                _audit("ride_request_fallback", mobile, f"{pickup}->{dropoff}", f"fare={fare_est}")
                return jsonify({"message":"ride_requested_immediate_fallback","fare_estimate": fare_est}), 201
            except Exception as e2:
                return jsonify({"error": err or "ride_create_failed", "wait_log": wait_log, "eta": eta, "fallback_error": str(e2)}), 500
        _audit("ride_request", mobile, f"{pickup}->{dropoff}", f"fare={fare_est}")
        return jsonify({"message":"ride_requested","fare_estimate": fare_est, "wait_log": wait_log, "eta": eta}), 201
    except Exception as e:
        try:
            run_query("INSERT INTO rides (mobile,pickup,dropoff,status,fare_estimate) VALUES (?,?,?,?,?)",
                      (mobile, pickup, dropoff, "requested", fare_est))
            _audit("ride_request_exception_fallback", mobile, f"{pickup}->{dropoff}", f"fare={fare_est}")
            return jsonify({"message":"ride_requested_exception_fallback","fare_estimate": fare_est}), 201
        except Exception as e2:
            return jsonify({"error":"ride_create_failed_both","detail": f"{e} | fallback: {e2}"}), 500


@app.route("/rides/status/<int:ride_id>", methods=["GET"])
def rides_status_public(ride_id):
    try:
        rows = run_query("SELECT id,mobile,pickup,dropoff,status,fare_estimate,timestamp FROM rides WHERE id=? LIMIT 1", (ride_id,), fetch=True)
        if not rows:
            return jsonify({"error":"not_found"}), 404
        r = rows[0]
        return jsonify({"id": r[0], "mobile": r[1], "pickup": r[2], "dropoff": r[3], "status": r[4], "fare_estimate": r[5], "timestamp": r[6]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/cherry/command", methods=["POST"])
def cherry_command_public():
    """
    Public Cherry endpoint: receives recognized text or JSON with `mobile` and `text`.
    Example body: { "mobile": "...", "text": "open wallet" }
    This routes simple commands and returns structured actions for client to perform.
    """
    d = request.json or {}
    mobile = d.get("mobile")
    text = (d.get("text") or "").strip().lower()
    if not mobile or not text:
        return jsonify({"error":"mobile and text required"}), 400

    # Basic command routing
    if "balance" in text or "wallet" in text:
        row = run_query("SELECT balance FROM users WHERE mobile=? LIMIT 1", (mobile,), fetch=True)
        bal = row[0][0] if row else 0.0
        return jsonify({"action":"wallet_balance","balance": bal})
    if "rewards" in text or "earned" in text:
        rows = run_query("SELECT amount,source,timestamp FROM rewards WHERE mobile=? ORDER BY id DESC LIMIT 5", (mobile,), fetch=True)
        items = [{"amount": r[0], "source": r[1], "timestamp": r[2]} for r in rows]
        return jsonify({"action":"recent_rewards","rewards": items})
    if "order" in text or "grocery" in text:
        return jsonify({"action":"open_orders_section"})
    if "ride" in text or "book" in text:
        return jsonify({"action":"open_rides_section"})
    if "play" in text or "next" in text:
        return jsonify({"action":"video_control","command": text})
    # fallback
    return jsonify({"action":"unknown","text": text})

@app.route("/debug/whichdb", methods=["GET"])
def debug_whichdb():
    import os, sqlite3
    # replicate how the database module might form path: use env DATABASE_URL fallback to db.sqlite3
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
# ------------------------ AI routing ------------------------
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

@app.route("/ai/humanhelper", methods=["POST"])
def ai_humanhelper():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    history = data.get("history", None)
    if not prompt:
        return jsonify({"error":"prompt required"}), 400

    premium = is_premium_user(request)

    # Non-premium: Gemini-free -> LLaMA (no OpenAI/DeepSeek)
    if not premium:
        ok_g, ans_g = gemini_free_chat(prompt)
        if ok_g:
            return jsonify({"provider": "gemini-free", "ok": True, "answer": ans_g}), 200
        # LLaMA fallback
        llama_ok, llama_ans = ask_humanhelper_llama_chat(prompt, history=history, n_predict=int(os.getenv("LLAMA_N_PREDICT", LLAMA_N_PREDICT)), timeout=int(os.getenv("LLAMA_TIMEOUT", LLAMA_TIMEOUT)))
        if llama_ok:
            return jsonify({"provider":"llama","ok":True,"answer":llama_ans}), 200
        return jsonify({"provider":"none","ok":False,"answer": ans_g or llama_ans or "No provider responded."}), 502

    # Premium flow: OpenAI -> DeepSeek -> LLaMA fallback
    if OPENAI_API_KEY:
        ok_oa, ans_oa = ask_openai_rest(prompt)
        if ok_oa:
            return jsonify({"provider":"openai","ok":True,"answer":ans_oa}), 200

    if DEEPSEEK_API_KEY:
        ok_ds, ans_ds = ask_deepseek(prompt)
        if ok_ds:
            return jsonify({"provider":"deepseek","ok":True,"answer":ans_ds}), 200

    # fallback to LLaMA
    llama_ok, llama_ans = ask_humanhelper_llama_chat(prompt, history=history, n_predict=int(os.getenv("LLAMA_N_PREDICT", LLAMA_N_PREDICT)), timeout=int(os.getenv("LLAMA_TIMEOUT", LLAMA_TIMEOUT)))
    if llama_ok:
        return jsonify({"provider":"llama","ok":True,"answer":llama_ans}), 200

    return jsonify({"provider":"none","ok":False,"answer":"All providers failed. Please try again later."}), 502

@app.route("/ai/ask", methods=["POST"])
def ai_ask():
    data = request.json or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error":"prompt required"}), 400
    # Try LLaMA first
    ok, ans = ask_llama(prompt, n_predict=int(data.get("n_predict", 128)))
    if ok:
        return jsonify({"provider":"llama","ok":True,"answer":ans}), 200
    return jsonify({"provider":"none","ok":False,"answer":ans}), 502

@app.route("/ai/status", methods=["GET"])
def ai_status():
    return jsonify({
        "openai_configured": bool(OPENAI_API_KEY),
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "gemini_free_configured": bool(GEMINI_REST_URL and GEMINI_API_KEY) or GEMINI_SDK_OK,
        "llama_server": LLAMA_SERVER_URL,
        "notes": {
            "non_premium": "Gemini-free (REST or SDK) -> LLaMA fallback. No paid remotes used for non-premium.",
            "premium": "OpenAI -> DeepSeek -> LLaMA fallback."
        }
    })

# ------------------------ GitHub endpoints ------------------------
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

# ------------------------ Run app ------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("Starting HumanHelper backend on port", port)
    print("LLAMA_SERVER_URL:", LLAMA_SERVER_URL)
    print("Non-premium provider: Gemini-free via GEMINI_REST_URL/GEMINI_API_KEY or SDK (if present)")
    print("Premium providers: OpenAI (OPENAI_API_KEY), DeepSeek (DEEPSEEK_API_KEY), then LLaMA fallback")
    # start Flask
    app.run(debug=True, host="0.0.0.0", port=port, threaded=False, use_reloader=False)

=======
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status":"ok","message":"Human Helper API is live"})

# example endpoint
@app.route("/echo", methods=["POST"])
def echo():
    data = request.json or {}
    return jsonify({"you_sent": data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000
