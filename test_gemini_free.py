#!/usr/bin/env python3
# test_gemini_free.py
import os, requests, time, json
GURL = os.getenv("GEMINI_REST_URL", "")
GKEY = os.getenv("GEMINI_API_KEY", "")
PROMPT = "Hello Gemini, quick test reply please."

if not (GURL and GKEY):
    print("GEMINI_REST_URL or GEMINI_API_KEY not set. Skipping.")
    raise SystemExit(2)

headers = {"Authorization": f"Bearer {GKEY}", "Content-Type":"application/json"}
body = {"prompt": PROMPT, "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS","200"))}
t0 = time.time()
r = requests.post(GURL, headers=headers, json=body, timeout=20)
print("HTTP", r.status_code, "time_ms", (time.time()-t0)*1000.0)
try:
    print(r.json())
except Exception:
    print(r.text[:1000])
