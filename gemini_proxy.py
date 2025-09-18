# gemini_proxy.py
from flask import Blueprint, request, jsonify
import os, requests, json, time, traceback

gemini_bp = Blueprint("gemini_proxy", __name__, url_prefix="/ai")

def _env(name, default=None):
    v = os.getenv(name, default)
    return v if v is not None else default

def _log(*args, **kwargs):
    try:
        print("[GEMINI_PROXY]", *args, **kwargs)
    except Exception:
        pass

@gemini_bp.route("/gemini-proxy", methods=["POST"])
def gemini_proxy():
    """
    Local proxy to call Google Generative Language REST endpoint.
    Header required:
      X-GEMINI-PROXY-KEY: <value matching GEMINI_PROXY_KEY in .env>

    Body (JSON): {"prompt": "your text"} (other fields ignored)

    Env vars used:
      GEMINI_PROXY_KEY, GEMINI_API_KEY, GEMINI_REST_URL, GEMINI_AUTH_METHOD (key|bearer),
      GEMINI_TIMEOUT, GEMINI_MAX_TOKENS, GEMINI_TEMPERATURE
    """
    try:
        # auth
        expected = (_env("GEMINI_PROXY_KEY") or "").strip()
        got = request.headers.get("X-GEMINI-PROXY-KEY", "")
        if not expected or got != expected:
            return jsonify({"ok": False, "error": "proxy auth failed"}), 403

        # input
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt") or data.get("text") or data.get("message") or ""
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"ok": False, "error": 'prompt (string) required in JSON body, e.g. {"prompt":"..."}'}), 400
        prompt = prompt.strip()

        # config
        rest_url = (_env("GEMINI_REST_URL") or "").strip()
        if not rest_url:
            return jsonify({"ok": False, "error": "GEMINI_REST_URL not configured"}), 500

        auth_method = (_env("GEMINI_AUTH_METHOD") or "bearer").lower()
        api_key = (_env("GEMINI_API_KEY") or "").strip()
        try:
            timeout = int(_env("GEMINI_TIMEOUT", "20"))
        except Exception:
            timeout = 20
        try:
            max_tokens = int(_env("GEMINI_MAX_TOKENS", "512"))
        except Exception:
            max_tokens = 512
        try:
            temperature = float(_env("GEMINI_TEMPERATURE", "0.3"))
        except Exception:
            temperature = 0.3

        headers = {"Content-Type": "application/json"}
        url = rest_url
        if auth_method == "key":
            if not api_key:
                return jsonify({"ok": False, "error": "GEMINI_API_KEY required for key auth"}), 500
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}key={api_key}"
        elif auth_method == "bearer":
            if not api_key:
                return jsonify({"ok": False, "error": "GEMINI_API_KEY required for bearer auth"}), 500
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            return jsonify({"ok": False, "error": f"unsupported GEMINI_AUTH_METHOD={auth_method}"}), 500

        # We'll attempt multiple payload shapes in order, to support different GenAI endpoint versions.
        # 1) Newer v1 generateContent style: {"contents":[{"role":"user","parts":[{"text":"..."}]}], ...}
        # 2) Alternative v1 style: {"prompt": {"text":"..."}, ...}
        # 3) v1beta2 style: {"instances":[{"content":"..."}], ...}
        # 4) Simple fallback: {"text": "..."}

        candidate_payloads = []

        # payload A: v1 generateContent 'contents' with parts (works for many v1 endpoints)
        candidate_payloads.append({
            "shape": "v1_contents_parts",
            "json": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                # these fields are accepted by many v1 endpoints (if unsupported server will error)
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        })

        # payload B: v1 'prompt' shape (some docs/examples use this)
        candidate_payloads.append({
            "shape": "v1_prompt_text",
            "json": {
                "prompt": {"text": prompt},
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        })

        # payload C: older v1beta2-like instances shape
        candidate_payloads.append({
            "shape": "v1beta2_instances",
            "json": {
                "instances": [{"content": prompt}],
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        })

        # payload D: minimal simple text body
        candidate_payloads.append({
            "shape": "minimal_text",
            "json": {"text": prompt}
        })

        last_error = None
        for p in candidate_payloads:
            try:
                _log("Trying payload shape:", p["shape"])
                # show only small preview
                try:
                    _log("URL:", url)
                    _log("HEADERS:", {k: (v if k != "Authorization" else "REDACTED") for k, v in headers.items()})
                    _log("PAYLOAD preview:", json.dumps(p["json"])[:1200])
                except Exception:
                    pass

                resp = requests.post(url, headers=headers, json=p["json"], timeout=timeout)
            except Exception as e:
                last_error = f"request error for shape {p['shape']}: {e}"
                _log(last_error)
                # try next payload shape
                time.sleep(0.1)
                continue

            body_text = ""
            try:
                body_text = resp.text or ""
                if len(body_text) > 2000:
                    body_text = body_text[:2000] + "..."
            except Exception:
                body_text = "<no body>"

            _log("Response status:", resp.status_code)
            _log("Response preview:", body_text)

            if resp.status_code >= 400:
                # record last error and try next payload shape
                last_error = f"shape={p['shape']} status={resp.status_code} body={body_text}"
                # If 401/403/404 with "Requested entity was not found" or "API key invalid" don't try other shapes
                # but we still capture the last_error and exit loop when appropriate:
                if resp.status_code in (401, 403):
                    return jsonify({"ok": False, "error": f"Gemini REST error: {resp.status_code} {body_text}"}), 502
                # try next shape
                time.sleep(0.1)
                continue

            # success (2xx) — attempt to parse JSON and extract text
            try:
                j = resp.json()
            except Exception:
                # non-json success — return raw text
                return jsonify({"provider": "gemini", "ok": True, "answer": resp.text}), 200

            # Attempt to find text/answer in common fields
            answer = None
            try:
                if isinstance(j, dict):
                    # candidates -> content -> parts -> text
                    if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
                        cand = j["candidates"][0]
                        cont = cand.get("content") or cand.get("message") or cand
                        if isinstance(cont, dict):
                            parts = cont.get("parts") or []
                            if parts and isinstance(parts, list):
                                p0 = parts[0]
                                if isinstance(p0, dict):
                                    answer = p0.get("text") or p0.get("content") or None
                        if not answer:
                            answer = cand.get("text") or cand.get("content") or None

                    # outputs
                    if not answer and "outputs" in j and isinstance(j["outputs"], list) and j["outputs"]:
                        o = j["outputs"][0]
                        if isinstance(o, dict):
                            # some outputs have nested content structure
                            answer = o.get("content") or o.get("text") or None
                        elif isinstance(o, str):
                            answer = o

                    # predictions
                    if not answer and "predictions" in j and isinstance(j["predictions"], list) and j["predictions"]:
                        p0 = j["predictions"][0]
                        if isinstance(p0, dict):
                            answer = p0.get("content") or p0.get("text") or None

                    # top-level fallbacks
                    if not answer:
                        for key in ("text", "response", "output", "result"):
                            v = j.get(key)
                            if isinstance(v, str) and v.strip():
                                answer = v
                                break

                    # if still not found, try some nested candidate["content"]["parts"] patterns
                    if not answer:
                        # try cand.content.parts[0].text deeper if cand earlier wasn't dict
                        try:
                            if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
                                cand = j["candidates"][0]
                                cont = cand.get("content") or {}
                                if isinstance(cont, dict):
                                    parts = cont.get("parts") or []
                                    if parts and isinstance(parts, list):
                                        p0 = parts[0]
                                        if isinstance(p0, dict):
                                            answer = p0.get("text") or p0.get("content") or None
                        except Exception:
                            pass

            except Exception as e:
                _log("Parsing exception:", e, traceback.format_exc())

            if answer and isinstance(answer, str) and answer.strip():
                return jsonify({"provider": "gemini", "ok": True, "answer": answer.strip(), "shape_used": p["shape"]}), 200

            # If we reached here, response was 2xx but we couldn't find a simple answer string.
            # Return the full JSON (trimmed) for debugging.
            trimmed = j
            return jsonify({"provider": "gemini", "ok": True, "answer": trimmed, "note": "full JSON returned (no obvious text field found)", "shape_used": p["shape"]}), 200

        # No payload shape succeeded
        if last_error:
            return jsonify({"ok": False, "error": f"All payload shapes failed. Last: {last_error}"}), 502
        return jsonify({"ok": False, "error": "All payload shapes failed; no request was successful."}), 502

    except Exception as e:
        _log("Unhandled exception:", e, traceback.format_exc())
        return jsonify({"ok": False, "error": f"internal proxy error: {e}"}), 500
