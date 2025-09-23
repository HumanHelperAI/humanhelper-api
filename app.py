#!/usr/bin/env python3
# app.py - HumanHelper consolidated backend (OpenRouter/free-first + fallbacks)
# Paste this whole file into your project (nano app.py) and save.

from __future__ import annotations
import os
import time
import threading
import random
import json
from functools import wraps
from typing import Tuple, Optional, Any, Dict, List

import requests
from flask import Flask, jsonify, request, Blueprint
from flask_cors import CORS

# load environment if .env present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------- Config / Environment --------------------
# Admin token for admin endpoints
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")

# Origins allowed for CORS (replace with your GitHub Pages origin + local)
ALLOWED_ORIGINS = [
    os.getenv("FRONTEND_ORIGIN", "https://humanhelperai.github.io"),
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

# External provider keys / endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_REST_URL = os.getenv("GEMINI_REST_URL", "")  # If you have a proxy/rest wrapper
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

# LLaMA local server fallback
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:5001/completion")

# Timeouts
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "20"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", DEFAULT_TIMEOUT))
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", DEFAULT_TIMEOUT))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "20"))
LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", "30"))

# Free-models priority (comma separated; try in order). Example env:
# FREE_MODELS_PRIORITY="openrouter:gpt-oss-20b,openrouter:gpt-oss-120b,openrouter:gpt-oss-20b-preview"
FREE_MODELS_PRIORITY = [s.strip() for s in os.getenv("FREE_MODELS_PRIORITY", "").split(",") if s.strip()]

# Optional explicit list of OpenRouter models in order
OPENROUTER_MODELS = [m.strip() for m in os.getenv("OPENROUTER_MODELS", "").split(",") if m.strip()]

# Deepseek model selection (if you want to prefer specific DeepSeek models)
DEEPSEEK_MODELS = [m.strip() for m in os.getenv("DEEPSEEK_MODELS", "").split(",") if m.strip()]

# GitHub
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# -------------------- Defensive local module stubs --------------------
# These local modules may exist in your project. If not, provide harmless stubs.
try:
    from database import init_db, run_query, cleanup_old_logs
except Exception:
    def init_db(*args, **kwargs):
        print("init_db(): database module missing; stub used")
    def run_query(sql, params=(), fetch=False):
        # return empty list for fetch
        return [] if fetch else None
    def cleanup_old_logs(): pass
    print("Warning: database module not found — using stubs")

try:
    from writer import start_writer, enqueue_write
except Exception:
    def start_writer(): pass
    def enqueue_write(sql, params=(), timeout=1.0):
        # emulate queued write: (ok, result?, wait_log, eta_seconds, error)
        return True, None, ["stub"], 0, None
    print("Warning: writer module not found — using stub enqueue_write")

try:
    from wallet import deposit as wallet_deposit, withdraw as wallet_withdraw
except Exception:
    def wallet_deposit(mobile, amount):
        return False, "wallet not configured"
    def wallet_withdraw(mobile, amount):
        return False, "wallet not configured"
    print("Warning: wallet module not found — using stubs")

try:
    from earnings import reward_user
except Exception:
    def reward_user(mobile, video_id, content_type, duration):
        return False, "earnings not configured"
    print("Warning: earnings module not found — using stub reward_user")

# -------------------- Flask app init --------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# helpful after_request headers
@app.after_request
def add_common_headers(resp):
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, X-Admin-Token, X-Premium"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

# -------------------- Network helper with retries --------------------
def _do_request_with_retries(method: str, url: str, headers: dict | None = None, json_body: Any = None,
                             timeout: int = DEFAULT_TIMEOUT, max_attempts: int = 3) -> Tuple[bool, Any]:
    attempt = 0
    base = 0.6
    last_exc = None
    while attempt < max_attempts:
        try:
            if method.lower() == "post":
                r = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            else:
                r = requests.get(url, headers=headers, params=json_body, timeout=timeout)
            # treat 2xx as success
            if 200 <= r.status_code < 300:
                return True, r
            if r.status_code in (429, 503):
                # respect Retry-After if present
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        wait = int(ra) + random.uniform(0.2, 0.6)
                    except Exception:
                        wait = base * (2 ** attempt) + random.uniform(0, 0.3)
                else:
                    wait = base * (2 ** attempt) + random.uniform(0, 0.3)
                time.sleep(wait)
                attempt += 1
                continue
            # other non-2xx - return for handling by caller
            return False, r
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(base * (2 ** attempt) + random.uniform(0, 0.3))
            attempt += 1
    return False, f"request failed after {max_attempts} attempts: {last_exc}"

# -------------------- OpenRouter (free models) wrapper --------------------
def ask_openrouter(prompt: str, model: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[bool, str]:
    """
    Sends a request to OpenRouter (or compatible API). Model must be the router model id used by openrouter.
    Example payload shape follows OpenRouter chat/completions: messages list.
    """
    if not OPENROUTER_API_KEY:
        return False, "OpenRouter API key not configured"
    url = OPENROUTER_URL
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role":"user","content":prompt}],
        # you may tune these via env vars if needed
        "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", "512"))
    }
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=timeout, max_attempts=3)
    if not ok:
        if isinstance(resp, requests.Response):
            return False, f"OpenRouter error {resp.status_code}: {resp.text[:800]}"
        return False, str(resp)
    try:
        j = resp.json()
        # OpenRouter returns choices -> message -> content (like OpenAI)
        choices = j.get("choices") or []
        if choices:
            first = choices[0]
            msg = first.get("message") or first.get("text") or {}
            if isinstance(msg, dict):
                txt = msg.get("content") or msg.get("text")
            else:
                txt = msg
            return True, (txt or "").strip()
        # fallback: try top-level fields
        for k in ("content", "text", "output"):
            if k in j and isinstance(j[k], str):
                return True, j[k].strip()
        return True, json.dumps(j)[:1000]
    except Exception as e:
        return False, f"OpenRouter parse error: {e}"

# -------------------- OpenAI (paid fallback) wrapper (REST) --------------------
def ask_openai_rest(prompt: str, timeout: int = OPENAI_TIMEOUT) -> Tuple[bool, str]:
    if not OPENAI_API_KEY:
        return False, "OpenAI not configured"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "512"))
    }
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=body, timeout=timeout)
    if not ok:
        if isinstance(resp, requests.Response):
            return False, f"OpenAI error {resp.status_code}: {resp.text[:800]}"
        return False, str(resp)
    try:
        j = resp.json()
        choices = j.get("choices", []) or []
        if choices:
            # chat-completion shape
            content = choices[0].get("message", {}).get("content") or choices[0].get("text")
            return True, (content or "").strip()
        return True, json.dumps(j)[:1000]
    except Exception as e:
        return False, f"OpenAI parse error: {e}"

# -------------------- DeepSeek wrapper --------------------
def ask_deepseek(prompt: str, model: Optional[str] = None, timeout: int = DEEPSEEK_TIMEOUT) -> Tuple[bool, str]:
    if not DEEPSEEK_API_KEY:
        return False, "DeepSeek not configured"
    url = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model or (DEEPSEEK_MODELS[0] if DEEPSEEK_MODELS else os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        "messages": [{"role":"user","content":prompt}],
        "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.4")),
        "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "512"))
    }
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=timeout)
    if not ok:
        if isinstance(resp, requests.Response):
            return False, f"DeepSeek error {resp.status_code}: {resp.text[:800]}"
        return False, str(resp)
    try:
        j = resp.json()
        choices = j.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content") or choices[0].get("text")
            return True, (content or "").strip()
        return True, json.dumps(j)[:1000]
    except Exception as e:
        return False, f"DeepSeek parse error: {e}"

# -------------------- Gemini-free (REST or SDK) wrapper --------------------
def gemini_free_chat(prompt: str, timeout: int = GEMINI_TIMEOUT) -> Tuple[bool, str]:
    # If you have a REST proxy for a free Gemini model, use it:
    if GEMINI_REST_URL and GEMINI_API_KEY:
        headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "512"))}
        ok, resp = _do_request_with_retries("post", GEMINI_REST_URL, headers=headers, json_body=payload, timeout=timeout)
        if not ok:
            if isinstance(resp, requests.Response):
                return False, f"Gemini-free error {resp.status_code}: {resp.text[:800]}"
            return False, str(resp)
        try:
            j = resp.json()
            for k in ("content", "text", "answer", "response"):
                if k in j and isinstance(j[k], str):
                    return True, j[k].strip()
            if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                first = j["choices"][0]
                for k in ("text", "content", "message"):
                    if k in first and isinstance(first[k], str):
                        return True, first[k].strip()
            return True, json.dumps(j)[:1000]
        except Exception as e:
            return False, f"Gemini-free parse error: {e}"
    # Optionally, if SDK available, attempt SDK usage (not required here)
    return False, "Gemini-free not configured"

# -------------------- LLaMA local fallback --------------------
def _parse_llama_response_obj(j: dict) -> Optional[str]:
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

def ask_llama(prompt: str, n_predict: int = 256, timeout: int = LLAMA_TIMEOUT) -> Tuple[bool, str]:
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
        txt = _parse_llama_response_obj(j)
        if txt:
            return True, txt
        return True, (r.text or "").strip()
    except Exception:
        return True, (r.text or "").strip() or ""

# -------------------- Provider orchestration: free-first -> fallbacks --------------------
def _try_free_models(prompt: str) -> Tuple[bool, str, str]:
    """
    Try to satisfy prompt using free-models list (OPENROUTER first), then Gemini-free,
    then LLaMA local. Returns (ok, answer, provider_name).
    """
    # 1) OpenRouter/Free models list (explicit FREE_MODELS_PRIORITY)
    # Format for each entry: "openrouter:model_name" or "openrouter:model_name:opt"
    for entry in (FREE_MODELS_PRIORITY or OPENROUTER_MODELS):
        if not entry:
            continue
        # support "openrouter:MODEL_ID" entries
        if entry.startswith("openrouter:"):
            model = entry.split(":", 1)[1].strip()
            if model and OPENROUTER_API_KEY:
                ok, ans = ask_openrouter(prompt, model=model, timeout=DEFAULT_TIMEOUT)
                if ok:
                    return True, ans, f"openrouter:{model}"
                # else log and continue
        else:
            # if entry is a bare model name, try as openrouter model
            model = entry
            if OPENROUTER_API_KEY:
                ok, ans = ask_openrouter(prompt, model=model, timeout=DEFAULT_TIMEOUT)
                if ok:
                    return True, ans, f"openrouter:{model}"

    # 2) Gemini-free (REST proxy)
    if GEMINI_REST_URL and GEMINI_API_KEY:
        ok, ans = gemini_free_chat(prompt)
        if ok:
            return True, ans, "gemini-free"

    # 3) local LLaMA
    ok, ans = ask_llama(prompt, timeout=LLAMA_TIMEOUT)
    if ok:
        return True, ans, "llama-local"

    return False, "No free provider responded", "none"

def _try_premium_providers(prompt: str) -> Tuple[bool, str, str]:
    """
    Try premium providers in order: OpenAI -> DeepSeek -> LLaMA fallback
    """
    # OpenAI
    if OPENAI_API_KEY:
        ok, ans = ask_openai_rest(prompt, timeout=OPENAI_TIMEOUT)
        if ok:
            return True, ans, "openai"

    # DeepSeek (if key)
    if DEEPSEEK_API_KEY:
        # try models in DEEPSEEK_MODELS first if set
        models_to_try = DEEPSEEK_MODELS or [os.getenv("DEEPSEEK_MODEL", "deepseek-chat")]
        for m in models_to_try:
            ok, ans = ask_deepseek(prompt, model=m)
            if ok:
                return True, ans, f"deepseek:{m}"
        # fallback attempt with default
        ok, ans = ask_deepseek(prompt)
        if ok:
            return True, ans, "deepseek"

    # Final fallback to LLaMA
    ok, ans = ask_llama(prompt, timeout=LLAMA_TIMEOUT)
    if ok:
        return True, ans, "llama-local"

    return False, "No premium provider responded", "none"

# -------------------- Authentication / Admin helpers --------------------
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

def admin_required(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        token = request.headers.get("X-Admin-Token", "") or ""
        if token != ADMIN_TOKEN and token != os.getenv("ADMIN_TOKEN", ADMIN_TOKEN):
            return jsonify({"error": "admin auth failed"}), 403
        return fn(*args, **kwargs)
    return inner

def _audit(action: str, actor: str, target: str, details: str):
    ip = request.headers.get("X-Forwarded-For") or request.remote_addr or "-"
    try:
        enqueue_write("INSERT INTO audit_log (action, actor, target, details, ip) VALUES (?,?,?,?,?)",
                      (action, actor, target, details, ip), timeout=3.0)
    except Exception:
        pass

# Admin endpoints (examples)
@admin_bp.get("/ping")
@admin_required
def admin_ping():
    return jsonify({"ok": True, "msg": "admin ping"}), 200

# -------------------- Basic public endpoints --------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Human Helper API is live", "status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    # attempt a cheap DB check if available (non-blocking)
    try:
        run_query("SELECT 1", fetch=True)
    except Exception:
        pass
    return jsonify({"message": "Human Helper API is live", "status": "ok"}), 200

# -------------------- User / wallet / transactions endpoints (stubs safe) --------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json or {}
    req_fields = ("name", "mobile", "aadhar", "password")
    if not all(data.get(k) for k in req_fields):
        return jsonify({"error": f"fields required: {', '.join(req_fields)}"}), 400
    try:
        ok, _, wait_log, eta, err = enqueue_write(
            "INSERT INTO users (name,mobile,aadhar,pan,password) VALUES (?,?,?,?,?)",
            (data.get("name"), data.get("mobile"), data.get("aadhar"), data.get("pan", ""), data.get("password")),
            timeout=8.0
        )
        if not ok:
            return jsonify({"error": err or "signup failed", "wait_log": wait_log, "eta_seconds": eta}), 400
        return jsonify({"message": "User registered", "wait_log": wait_log, "eta_seconds": eta}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    d = request.json or {}
    rows = run_query("SELECT balance, is_banned FROM users WHERE mobile=? AND password=?", (d.get("mobile"), d.get("password")), fetch=True)
    if rows:
        if rows[0][1]:
            return jsonify({"error": "Account banned"}), 403
        return jsonify({"message": "Login successful", "balance": rows[0][0]})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/balance/<mobile>", methods=["GET"])
def balance(mobile):
    r = run_query("SELECT balance FROM users WHERE mobile=?", (mobile,), fetch=True)
    return jsonify({"mobile": mobile, "balance": r[0][0] if r else 0.0})

@app.route("/deposit", methods=["POST"])
def deposit_route():
    d = request.json or {}
    try:
        ok, msg = wallet_deposit(d.get("mobile"), float(d.get("amount")))
        if not ok:
            return jsonify({"error": msg}), 400
        return jsonify({"mobile": d.get("mobile"), "message": msg}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/withdraw", methods=["POST"])
def withdraw_route():
    d = request.json or {}
    try:
        ok, msg = wallet_withdraw(d.get("mobile"), float(d.get("amount")))
        if not ok:
            return jsonify({"error": msg}), 400
        return jsonify({"mobile": d.get("mobile"), "message": msg}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------- AI endpoints --------------------
def is_premium_user(req: request.__class__) -> bool:
    try:
        j = req.get_json(silent=True) or {}
        if isinstance(j.get("premium"), bool):
            return j.get("premium")
    except Exception:
        pass
    h = (req.headers.get("X-Premium") or "").strip().lower()
    if h in ("1", "true", "yes", "on"):
        return True
    # Admin token counts as premium for convenience
    admin_token = req.headers.get("X-Admin-Token", "")
    if admin_token and admin_token == ADMIN_TOKEN:
        return True
    return False

@app.route("/ai/humanhelper", methods=["POST"])
def ai_humanhelper():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    premium = is_premium_user(request)

    # If free-first flow (try OpenRouter free models + Gemini-free + LLaMA)
    ok_free, ans_free, provider_free = _try_free_models(prompt)
    if ok_free and not premium:
        return jsonify({"provider": provider_free, "ok": True, "answer": ans_free}), 200

    # If request is premium, or free attempts failed and we allow paid fallbacks:
    # Try premium providers (OpenAI -> DeepSeek -> LLaMA)
    ok_prem, ans_prem, provider_prem = _try_premium_providers(prompt)
    if ok_prem:
        return jsonify({"provider": provider_prem, "ok": True, "answer": ans_prem}), 200

    # If premium failed but free attempt exists, return it as best-effort
    if ok_free:
        return jsonify({"provider": provider_free, "ok": True, "answer": ans_free, "note": "served from best available free provider"}), 200

    # Final fallback: respond with failure
    return jsonify({"provider": "none", "ok": False, "answer": "All providers failed. Please try again later."}), 502

@app.route("/ai/ask", methods=["POST"])
def ai_ask():
    data = request.json or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    # If user sets "provider" in body, obey when possible
    provider = (data.get("provider") or "").strip().lower()
    if provider.startswith("openrouter:"):
        model = provider.split(":", 1)[1]
        ok, ans = ask_openrouter(prompt, model=model)
        return (jsonify({"provider": f"openrouter:{model}", "ok": ok, "answer": ans}), 200 if ok else 502)
    if provider == "openai":
        ok, ans = ask_openai_rest(prompt)
        return (jsonify({"provider": "openai", "ok": ok, "answer": ans}), 200 if ok else 502)
    if provider == "deepseek":
        ok, ans = ask_deepseek(prompt)
        return (jsonify({"provider": "deepseek", "ok": ok, "answer": ans}), 200 if ok else 502)
    if provider in ("gemini", "gemini-free"):
        ok, ans = gemini_free_chat(prompt)
        return (jsonify({"provider": "gemini-free", "ok": ok, "answer": ans}), 200 if ok else 502)
    # default: try free -> premium
    ok_free, ans_free, prov_free = _try_free_models(prompt)
    if ok_free:
        return jsonify({"provider": prov_free, "ok": True, "answer": ans_free}), 200
    ok_prem, ans_prem, prov_prem = _try_premium_providers(prompt)
    if ok_prem:
        return jsonify({"provider": prov_prem, "ok": True, "answer": ans_prem}), 200
    return jsonify({"provider": "none", "ok": False, "answer": "No provider responded."}), 502

@app.route("/ai/status", methods=["GET"])
def ai_status():
    return jsonify({
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "gemini_configured": bool(GEMINI_REST_URL and GEMINI_API_KEY),
        "llama_server": LLAMA_SERVER_URL,
        "free_models_priority": FREE_MODELS_PRIORITY or OPENROUTER_MODELS,
        "notes": {
            "free_first": "Tries OpenRouter free models, then Gemini-free then LLaMA local. Premium providers used if configured or when premium requests.",
            "premium_flow": "OpenAI -> DeepSeek -> LLaMA fallback"
        }
    })

# -------------------- GitHub helpers --------------------
def github_get_file(owner: str, repo: str, path: str, ref: str = "main") -> Tuple[bool, str]:
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not configured"
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.raw"}
    ok, resp = _do_request_with_retries("get", url, headers=headers, timeout=15)
    if not ok:
        return False, f"GitHub fetch failed: {resp}"
    return True, resp.text

def github_create_gist(files: dict, description: str = "", public: bool = False) -> Tuple[bool, str]:
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not configured"
    url = "https://api.github.com/gists"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Content-Type": "application/json"}
    payload = {"files": {name: {"content": content} for name, content in files.items()},
               "description": description, "public": public}
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=15)
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
        return jsonify({"error": "owner,repo,path required"}), 400
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
        return jsonify({"error": "files dict required"}), 400
    ok, res = github_create_gist(files, description=desc, public=public)
    if not ok:
        return jsonify({"error": res}), 400
    return jsonify({"ok": True, "url": res})

# -------------------- Register admin blueprint and background tasks --------------------
try:
    app.register_blueprint(admin_bp)
except Exception as e:
    print("Warning: failed to register admin_bp:", e)

# background cleanup thread (non-blocking)
def _cleanup_loop():
    while True:
        try:
            cleanup_old_logs()
        except Exception:
            # safe ignore
            pass
        time.sleep(6 * 60 * 60)

threading.Thread(target=_cleanup_loop, daemon=True).start()

# init db/writer if present (non-fatal)
try:
    init_db()
except Exception as e:
    print("init_db failed (nonfatal):", e)
try:
    start_writer()
except Exception as e:
    print("start_writer failed (nonfatal):", e)

# -------------------- Run server (local dev) --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("Starting HumanHelper backend on port", port)
    print("Free models priority:", FREE_MODELS_PRIORITY or OPENROUTER_MODELS)
    app.run(host="0.0.0.0", port=port, debug=True)
