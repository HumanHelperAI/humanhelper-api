#!/usr/bin/env python3
# test_openrouter_models.py
# Usage:
#   export OPENROUTER_API_KEY="sk-..."
#   export OPENROUTER_URL="https://openrouter.ai/api/v1/chat/completions"
#   export FREE_MODELS_PRIORITY="modelA,modelB,modelC"   # comma separated (model ids)
#   python test_openrouter_models.py

import os, sys, time, json, csv, requests
from datetime import datetime

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
MODELS_ENV = os.getenv("FREE_MODELS_PRIORITY", "")
OUT_CSV = os.path.expanduser("~/humanhelper/backend/openrouter_test_results.csv")
TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "20"))

if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set in env.")
    sys.exit(2)

models = [m.strip() for m in MODELS_ENV.split(",") if m.strip()]
if not models:
    print("ERROR: FREE_MODELS_PRIORITY not set or empty. Set env to comma-separated list of model ids.")
    sys.exit(2)

prompt = os.getenv("OPENROUTER_TEST_PROMPT", "Hello, please provide a short test reply. (HumanHelper test)")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    # optional: "HTTP-Referer": "<your-site>", "X-Title": "<name>"
}

def try_model(model_name):
    body = {
        "model": model_name,
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", "200")),
        "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))
    }
    t0 = time.time()
    try:
        r = requests.post(URL, headers=headers, json=body, timeout=TIMEOUT)
        dt = (time.time()-t0)*1000.0
        status = r.status_code
        text = ""
        try:
            j = r.json()
            # try common shapes
            if isinstance(j, dict):
                if "choices" in j and j["choices"]:
                    first = j["choices"][0]
                    if isinstance(first, dict):
                        msg = first.get("message") or first
                        text = msg.get("content") or msg.get("text") or ""
                for k in ("text","content","answer","response"):
                    if not text and k in j and isinstance(j[k], str):
                        text = j[k]
            if not text and isinstance(j, str):
                text = j
        except Exception:
            text = (r.text or "")[:800]
        return True, status, dt, (text.strip()[:600] if text else "")
    except requests.exceptions.RequestException as e:
        dt = (time.time()-t0)*1000.0
        return False, None, dt, str(e)[:800]

# ensure output CSV has header if missing
first_write = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    if first_write:
        writer.writerow(["timestamp","model","ok","http_status","time_ms","excerpt_or_error"])
    for m in models:
        print(f"[{datetime.utcnow().isoformat()}] Testing model: {m} ...", flush=True)
        ok, status, dt, excerpt = try_model(m)
        writer.writerow([datetime.utcnow().isoformat(), m, "yes" if ok else "no", status if status else "", f"{dt:.1f}", excerpt])
        fh.flush()
        print(" ->", "OK" if ok else "FAIL", "status:", status, "time_ms:", f"{dt:.1f}", "excerpt:", excerpt[:200])
        # small pause between calls
        time.sleep(0.6)

print("Done. Results appended to:", OUT_CSV)
