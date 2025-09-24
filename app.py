#!/usr/bin/env python3
# app.py - HumanHelper consolidated backend (OpenRouter-first + fallbacks + logging)
# Paste this entire file into nano app.py and save.

import os
import time
import threading
import logging
import random
import json
from functools import wraps
from typing import Tuple, Optional

import requests
from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from dotenv import load_dotenv

# Load .env if present (local dev)
load_dotenv()

# ------------------------ Logging ------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("humanhelper")

# ------------------------ Defensive local modules ------------------------
try:
    from database import init_db, run_query, cleanup_old_logs
except Exception:
    def init_db(*a, **k): pass
    def run_query(sql, params=(), fetch=False):
        if fetch:
            return []
        return None
    def cleanup_old_logs(): pass
    logger.warning("database module not found — using stubs")

try:
    from writer import start_writer, enqueue_write
except Exception:
    def start_writer(): pass
    def enqueue_write(sql, params=(), timeout=1.0):
        return True, None, ["immediate"], 0, None
    logger.warning("writer module not found — using stub enqueue_write")

try:
    from wallet import deposit, withdraw
except Exception:
    def deposit(mobile, amount): return False, "wallet not configured"
    def withdraw(mobile, amount): return False, "wallet not configured"
    logger.warning("wallet module not found — using stubs")

try:
    from earnings import reward_user
except Exception:
    def reward_user(mobile, video_id, content_type, duration): return False, "earnings not configured"
    logger.warning("earnings module not found — using stub reward_user")

# ------------------------ Config / env ------------------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_REST_URL = os.getenv("GEMINI_REST_URL", "")
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:5001/completion")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://api.openrouter.ai/v1/chat/completions")
FREE_MODELS_ENV = os.getenv("FREE_MODELS_PRIORITY", "")  # comma-separated list of model identifiers
FREE_MODELS_PRIORITY = [m.strip() for m in FREE_MODELS_ENV.split(",") if m.strip()]
GEMINI_SDK_OK = False
GEMINI_SDK = None

# Try optionally import google generative SDK
try:
    import google.generativeai as genai
    GEMINI_SDK = genai
    GEMINI_SDK_OK = bool(GEMINI_API_KEY)
    if GEMINI_SDK_OK:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini SDK configured")
except Exception:
    GEMINI_SDK_OK = False

# Timeouts
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "20"))
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "20"))
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "20"))
LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", "30"))
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "20"))

# ------------------------ HTTP helper with retries ------------------------
def _do_request_with_retries(method: str, url: str, headers: dict = None, json_body: dict = None,
                             timeout: int = 15, max_attempts: int = 3) -> Tuple[bool, object]:
    attempt = 0
    base = 0.6
    last_exc = None
    while attempt < max_attempts:
        try:
            if method.lower() == "post":
                r = requests.post(url, headers=headers or {}, json=json_body, timeout=timeout)
            else:
                r = requests.get(url, headers=headers or {}, params=json_body, timeout=timeout)
            # Log short response for debugging (only first attempt or errors)
            if 200 <= r.status_code < 300:
                return True, r
            # Retry on 429/503 with backoff
            if r.status_code in (429, 503):
                wait = base * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("Transient %s from %s: status=%s. Retrying after %.1fs", method.upper(), url, r.status_code, wait)
                time.sleep(wait)
                attempt += 1
                continue
            # other non-2xx -> return False with response
            return False, r
        except requests.exceptions.RequestException as e:
            last_exc = e
            wait = base * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning("RequestException calling %s: %s (retrying in %.1fs)", url, e, wait)
            time.sleep(wait)
            attempt += 1
    return False, f"request failed after {max_attempts} attempts: {last_exc}"

# ------------------------ Provider wrappers ------------------------
def _log_provider_error(provider: str, resp_obj):
    """Log provider failure with status/body (safe truncated)."""
    try:
        if isinstance(resp_obj, requests.Response):
            body = (resp_obj.text or "")[:1000]
            logger.error("Provider %s failed: status=%s body=%s", provider, resp_obj.status_code, body)
        else:
            logger.error("Provider %s failed: %s", provider, str(resp_obj))
    except Exception:
        logger.exception("While logging provider error for %s", provider)

# OpenRouter (free-models-first)
def ask_openrouter_model(model_name: str, prompt: str, timeout: int = OPENROUTER_TIMEOUT) -> Tuple[bool, str]:
    if not OPENROUTER_API_KEY or not OPENROUTER_URL:
        return False, "openrouter not configured"
    url = OPENROUTER_URL
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role":"user","content": prompt}],
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", "400")),
        "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))
    }
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=timeout)
    if not ok:
        _log_provider_error(f"openrouter:{model_name}", resp)
        return False, f"openrouter {model_name} failed"
    try:
        j = resp.json()
        # typical OpenRouter / OpenAI-like shape
        choices = j.get("choices") or []
        if choices:
            first = choices[0]
            text = (first.get("message") or {}).get("content") or first.get("text")
            if text:
                logger.info("OpenRouter model %s responded OK", model_name)
                return True, text.strip()
        # fallback: top-level content keys
        for k in ("content","response","result","answer"):
            if k in j and isinstance(j[k], str):
                return True, j[k].strip()
        return False, "openrouter parse error"
    except Exception as e:
        _log_provider_error(f"openrouter:{model_name}", e)
        return False, str(e)

# OpenAI REST wrapper (fallback / premium)
def ask_openai_rest(prompt: str, timeout: int = OPENAI_TIMEOUT) -> Tuple[bool, str]:
    key = OPENAI_API_KEY
    if not key:
        return False, "openai not configured"
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
        _log_provider_error("openai", resp)
        return False, f"openai error"
    try:
        j = resp.json()
        choices = j.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content") or choices[0].get("text")
            return True, (text or "").strip()
        return False, "openai parse error"
    except Exception as e:
        _log_provider_error("openai", e)
        return False, str(e)

# DeepSeek wrapper
def ask_deepseek(prompt: str, timeout: int = DEEPSEEK_TIMEOUT) -> Tuple[bool, str]:
    key = DEEPSEEK_API_KEY
    if not key or not DEEPSEEK_URL:
        return False, "deepseek not configured"
    url = DEEPSEEK_URL
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS","400")),
        "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE","0.6"))
    }
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=timeout)
    if not ok:
        _log_provider_error("deepseek", resp)
        return False, "deepseek error"
    try:
        j = resp.json()
        choices = j.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content") or choices[0].get("text")
            return True, (text or "").strip()
        return False, "deepseek parse error"
    except Exception as e:
        _log_provider_error("deepseek", e)
        return False, str(e)

# Gemini-free wrapper (REST proxy if configured, else SDK)
def gemini_free_chat(prompt: str, timeout: int = GEMINI_TIMEOUT) -> Tuple[bool, str]:
    if GEMINI_REST_URL and GEMINI_API_KEY:
        headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS","400"))}
        ok, resp = _do_request_with_retries("post", GEMINI_REST_URL, headers=headers, json_body=payload, timeout=timeout)
        if not ok:
            _log_provider_error("gemini-free", resp)
            return False, "gemini-free error"
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
            return False, "gemini parse error"
        except Exception as e:
            _log_provider_error("gemini-free", e)
            return False, str(e)
    if GEMINI_SDK_OK and GEMINI_SDK:
        try:
            model = GEMINI_SDK.GenerativeModel("gemini")
            resp = model.generate_content(prompt)
            return True, (resp.text or "").strip()
        except Exception as e:
            _log_provider_error("gemini-sdk", e)
            return False, str(e)
    return False, "gemini not configured"

# LLaMA local server wrapper
def ask_llama(prompt: str, n_predict: int = 256, timeout: int = LLAMA_TIMEOUT) -> Tuple[bool, str]:
    url = LLAMA_SERVER_URL
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt, "n_predict": int(n_predict)}
    ok, resp = _do_request_with_retries("post", url, headers=headers, json_body=payload, timeout=timeout)
    if not ok:
        _log_provider_error("llama", resp)
        return False, f"llama error"
    try:
        j = resp.json()
        # parse heuristics
        for k in ("content","text","result","answer","output","response"):
            if k in j and isinstance(j[k], str):
                return True, j[k].strip()
        choices = j.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            for k in ("text","content","message"):
                v = first.get(k)
                if isinstance(v, str) and v.strip():
                    return True, v.strip()
        return True, (resp.text or "").strip()
    except Exception as e:
        _log_provider_error("llama", e)
        return False, str(e)

# ------------------------ App & CORS ------------------------
app = Flask(__name__)
ALLOWED_ORIGINS = [
    os.getenv("GITHUB_PAGES_ORIGIN", "https://humanhelperai.github.io"),
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# helper to add consistent CORS headers (optional)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, X-Admin-Token, X-Premium"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# ------------------------ Utilities ------------------------
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

# ------------------------ Admin blueprint ------------------------
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        token = request.headers.get("X-Admin-Token", "") or ""
        if token != ADMIN_TOKEN:
            return jsonify({"error":"admin auth failed"}), 403
        return fn(*a, **k)
    return wrapper

# some admin routes (examples)
@admin_bp.get("/ping")
@admin_required
def admin_ping():
    return jsonify({"ok": True, "who": "admin"}), 200

# register blueprint later in init section

# ------------------------ Simple health endpoints ------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Human Helper API is live", "status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    # minimal lightweight check - DB optional
    try:
        run_query("SELECT 1", fetch=True)
    except Exception:
        pass
    return jsonify({"message": "Human Helper API is live", "status": "ok"}), 200

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
    token = req.headers.get("X-Admin-Token", "")
    if token and token == ADMIN_TOKEN:
        return True
    return False

def try_free_models_first(prompt: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Try to use models in FREE_MODELS_PRIORITY via OpenRouter (or other mapping).
    Returns (ok, answer_or_none, provider_name_or_none)
    """
    if not FREE_MODELS_PRIORITY or not OPENROUTER_API_KEY:
        return False, None, None
    for model in FREE_MODELS_PRIORITY:
        # model might be a simple alias; attempt call
        try:
            ok, ans = ask_openrouter_model(model, prompt)
            if ok:
                return True, ans, f"openrouter:{model}"
            else:
                logger.info("OpenRouter model %s did not return (ok=%s)", model, ok)
        except Exception as e:
            logger.exception("Exception while calling openrouter model %s: %s", model, e)
    return False, None, None

@app.route("/ai/humanhelper", methods=["POST"])
def ai_humanhelper():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    history = data.get("history", None)
    if not prompt:
        return jsonify({"error":"prompt required"}), 400

    premium = is_premium_user(request)

    # Non-premium: try free models (OpenRouter) first, then Gemini-free, then local LLaMA
    if not premium:
        ok, ans, prov = try_free_models_first(prompt)
        if ok:
            logger.info("Non-premium: answered by %s", prov)
            return jsonify({"provider": prov, "ok": True, "answer": ans}), 200

        # try gemini free
        ok_g, ans_g = gemini_free_chat(prompt)
        if ok_g:
            logger.info("Non-premium: answered by gemini-free")
            return jsonify({"provider": "gemini-free", "ok": True, "answer": ans_g}), 200

        # try local llama fallback
        llama_ok, llama_ans = ask_llama(prompt, timeout=LLAMA_TIMEOUT)
        if llama_ok:
            logger.info("Non-premium: answered by llama local")
            return jsonify({"provider":"llama","ok":True,"answer":llama_ans}), 200

        logger.warning("Non-premium: no provider responded")
        return jsonify({"provider":"none","ok":False,"answer":"No provider responded."}), 502

    # Premium flow: prefer OpenAI -> DeepSeek -> OpenRouter free models (if desired) -> LLaMA
    if OPENAI_API_KEY:
        ok_oa, ans_oa = ask_openai_rest(prompt)
        if ok_oa:
            logger.info("Premium: answered by openai")
            return jsonify({"provider":"openai","ok":True,"answer":ans_oa}), 200
        else:
            logger.warning("OpenAI did not return an answer or failed")

    if DEEPSEEK_API_KEY and DEEPSEEK_URL:
        ok_ds, ans_ds = ask_deepseek(prompt)
        if ok_ds:
            logger.info("Premium: answered by deepseek")
            return jsonify({"provider":"deepseek","ok":True,"answer":ans_ds}), 200
        else:
            logger.warning("DeepSeek did not return an answer or failed")

    # Optionally try openrouter free models even for premium if configured
    ok, ans, prov = try_free_models_first(prompt)
    if ok:
        logger.info("Premium fallback: answered by %s", prov)
        return jsonify({"provider": prov, "ok": True, "answer": ans}), 200

    # fallback to llama
    llama_ok, llama_ans = ask_llama(prompt, timeout=LLAMA_TIMEOUT)
    if llama_ok:
        logger.info("Premium fallback: answered by llama")
        return jsonify({"provider":"llama","ok":True,"answer":llama_ans}), 200

    logger.error("All providers failed for premium request")
    return jsonify({"provider":"none","ok":False,"answer":"All providers failed. Please try again later."}), 502

# Short generic ask endpoint (tries LLaMA then OpenRouter then OpenAI)
@app.route("/ai/ask", methods=["POST"])
def ai_ask():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error":"prompt required"}), 400

    # try LLaMA first (local)
    ok, ans = ask_llama(prompt, n_predict=int(data.get("n_predict", 128)))
    if ok:
        return jsonify({"provider":"llama","ok":True,"answer":ans}), 200

    # try free models priority
    ok, ans, prov = try_free_models_first(prompt)
    if ok:
        return jsonify({"provider": prov, "ok": True, "answer": ans}), 200

    # try openai last
    ok_oa, ans_oa = ask_openai_rest(prompt)
    if ok_oa:
        return jsonify({"provider":"openai","ok":True,"answer":ans_oa}), 200

    return jsonify({"provider":"none","ok":False,"answer":"No provider responded."}), 502

@app.route("/ai/status", methods=["GET"])
def ai_status():
    return jsonify({
        "openai_configured": bool(OPENAI_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY and FREE_MODELS_PRIORITY),
        "deepseek_configured": bool(DEEPSEEK_API_KEY and DEEPSEEK_URL),
        "gemini_configured": bool(GEMINI_API_KEY or GEMINI_SDK_OK),
        "free_models_priority": FREE_MODELS_PRIORITY,
        "llama_server": LLAMA_SERVER_URL,
        "notes": {
            "free_first": "Tries OpenRouter free models, then Gemini-free then LLaMA local. Premium providers used if configured or when premium requests.",
            "premium_flow": "OpenAI -> DeepSeek -> LLaMA fallback"
        }
    })

# ------------------------ Example user/admin endpoints kept minimal ------------------------
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
        logger.exception("signup error")
        return jsonify({"error": str(e)}), 400

# (other routes like login, balance, deposit, withdraw, transactions, charity_balance, orders, rides, cherry etc.)
# For brevity, keep those you already had in your repo — you can paste them below if needed.

# ------------------------ Initialization ------------------------
def _cleanup_loop():
    while True:
        try:
            cleanup_old_logs()
        except Exception:
            # still keep sleeping even if no cleanup_old_logs defined
            pass
        time.sleep(6 * 60 * 60)

threading.Thread(target=_cleanup_loop, daemon=True).start()

# Register admin blueprint (safe)
try:
    app.register_blueprint(admin_bp)
except Exception as e:
    logger.warning("admin blueprint not registered: %s", e)

try:
    init_db()
except Exception as e:
    logger.warning("init_db() failed (nonfatal): %s", e)

try:
    start_writer()
except Exception as e:
    logger.warning("start_writer() failed (nonfatal): %s", e)

# ------------------------ Run (local dev) ------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info("Starting HumanHelper backend on port %s", port)
    logger.info("Free models priority: %s", FREE_MODELS_PRIORITY)
    app.run(host="0.0.0.0", port=port, debug=(os.getenv("FLASK_ENV")!="production"))
