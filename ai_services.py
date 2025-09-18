# ai_services.py
"""
Small adapter layer for AI providers used by app.py.
Provides ask_ai(provider, prompt) -> dict with keys:
  { "provider": "...", "ok": bool, "answer": "..." }
This file reads keys from environment. Keeps calls simple & defensive.
"""

import os
import json
import requests

# Try import OpenAI SDK if available
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# Try import Google Gemini SDK (if installed/works on device)
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# ----------------- OpenAI -----------------
def ask_openai(prompt: str, timeout: int = 10):
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or os.getenv("OPENAI")
    if not key:
        return {"provider": "openai", "ok": False, "answer": "OpenAI key not set on server"}
    if not HAS_OPENAI:
        # Best-effort: use simple HTTP fallback to the OpenAI REST API if 'openai' SDK not present
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 200
                },
                timeout=timeout
            )
            j = resp.json()
            text = j["choices"][0]["message"]["content"].strip() if "choices" in j and j["choices"] else j.get("error", {}).get("message", str(j))
            return {"provider": "openai", "ok": True, "answer": text}
        except Exception as e:
            return {"provider": "openai", "ok": False, "answer": "OpenAI request failed: " + str(e)}

    # Using openai SDK
    try:
        openai.api_key = key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            request_timeout=timeout
        )
        answer = resp["choices"][0]["message"]["content"].strip()
        return {"provider": "openai", "ok": True, "answer": answer}
    except Exception as e:
        return {"provider": "openai", "ok": False, "answer": "OpenAI error: " + str(e)}

# ----------------- Gemini -----------------
def ask_gemini(prompt: str, timeout: int = 10):
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return {"provider": "gemini", "ok": False, "answer": "Gemini key not set on server"}
    if not HAS_GEMINI:
        # If google.generativeai not installed or not usable on device, return informative message
        return {"provider": "gemini", "ok": False, "answer": "Gemini not available on this device or SDK not installed"}
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-pro")
        result = model.generate_content(prompt)
        text = (result.text or "").strip()
        return {"provider": "gemini", "ok": True, "answer": text}
    except Exception as e:
        return {"provider": "gemini", "ok": False, "answer": "Gemini error: " + str(e)}

# ----------------- DeepSeek (example HTTP wrapper) -----------------
def ask_deepseek(prompt: str, timeout: int = 8):
    """
    Placeholder wrapper for a 'DeepSeek' API.
    Many 3rd-party search/embedding services use a simple HTTP POST.
    Adjust URL/headers/body to your actual DeepSeek spec.
    """
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
    if not key:
        return {"provider": "deepseek", "ok": False, "answer": "DeepSeek not configured on server"}
    # Example URL — replace with real endpoint if you have it
    url = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.example/v1/query")
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"query": prompt, "top_k": 5},
            timeout=timeout
        )
        j = resp.json()
        # Try to extract text from response in a robust way:
        if isinstance(j, dict):
            text = j.get("answer") or j.get("result") or json.dumps(j)[:1000]
        else:
            text = str(j)
        return {"provider": "deepseek", "ok": True, "answer": str(text)}
    except Exception as e:
        return {"provider": "deepseek", "ok": False, "answer": "DeepSeek request failed: " + str(e)}

# ----------------- Dispatcher -----------------
def ask_ai(provider: str, prompt: str):
    provider = (provider or "openai").lower()
    prompt = (prompt or "").strip()
    if not prompt:
        return {"provider": provider, "ok": False, "answer": "prompt required"}

    if provider == "openai":
        return ask_openai(prompt)
    if provider == "gemini":
        return ask_gemini(prompt)
    if provider == "deepseek" or provider == "deepseek":
        return ask_deepseek(prompt)
    return {"provider": provider, "ok": False, "answer": "Invalid provider"}
