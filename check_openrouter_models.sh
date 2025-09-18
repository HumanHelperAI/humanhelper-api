#!/usr/bin/env bash
set -euo pipefail

# Location & settings
OUT="./openrouter_model_check_results.csv"
URL="${OPENROUTER_URL:-https://openrouter.ai/api/v1/chat/completions}"
API_KEY="${OPENROUTER_API_KEY:-}"
MODELS="${FREE_MODELS_PRIORITY:-deepseek/deepseek-v3.1-free,deepseek/r1-free,deepseek/qwen3-8b-free,nous/deephermes-3-llama-3-8b,openai/gpt-oss-20b,openai/gpt-oss-120b,google/gemini-2.0-flash-exp}"

if [ -z "$API_KEY" ]; then
  echo "ERROR: OPENROUTER_API_KEY is not set. Export it first:"
  echo "  export OPENROUTER_API_KEY=\"sk-or-...\""
  exit 2
fi

# CSV header
echo "timestamp,model,http_status,time_ms,ok,excerpt" > "$OUT"

IFS=',' read -r -a arr <<< "$MODELS"
for model in "${arr[@]}"; do
  model="$(echo -n "$model" | xargs)"   # trim spaces
  echo "[$(date -Iseconds)] Testing model: $model"

  ATTEMPTS=0
  MAX_ATTEMPTS=3
  OK=false
  HTTP_STATUS=""
  TIME_MS=0
  EXCERPT=""

  while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    ATTEMPTS=$((ATTEMPTS+1))
    T0=$(date +%s%3N)
    # Minimal valid payload for chat completions on OpenRouter
    PAYLOAD=$(cat <<JSON
{"model":"$model","messages":[{"role":"user","content":"Ping"}],"max_tokens":40}
JSON
)
    # Run the request (capture status + body)
    RESP=$(curl -sS -w "\n%{http_code}" -X POST "$URL" \
      -H "Authorization: Bearer $API_KEY" \
      -H "Content-Type: application/json" \
      -d "$PAYLOAD" --max-time 20) || true

    # separate body and status
    HTTP_STATUS=$(echo -n "$RESP" | tail -n1)
    BODY=$(echo -n "$RESP" | sed '$d' || true)
    T1=$(date +%s%3N)
    TIME_MS=$((T1 - T0))

    # determine OK if 2xx
    if [[ "$HTTP_STATUS" =~ ^2[0-9][0-9]$ ]]; then
      OK=true
      # extract short excerpt: message/choices content or first 200 chars
      EXCERPT=$(echo "$BODY" | tr -d '\n' | sed 's/"/'"'"'/g' | awk '{print substr($0,1,200)}')
      break
    fi

    # on 401/403 => stop (auth/permission); no retry
    if [[ "$HTTP_STATUS" == "401" || "$HTTP_STATUS" == "403" || "$HTTP_STATUS" == "404" ]]; then
      EXCERPT=$(echo "$BODY" | tr -d '\n' | sed 's/"/'"'"'/g' | awk '{print substr($0,1,400)}')
      break
    fi

    # transient codes: retry
    if [[ "$HTTP_STATUS" == "429" || "$HTTP_STATUS" == "503" || "$HTTP_STATUS" == "502" ]]; then
      echo "  transient ($HTTP_STATUS). attempt $ATTEMPTS/$MAX_ATTEMPTS -> retry after backoff"
      sleep $((ATTEMPTS * 2))
      continue
    fi

    # other non-200: record body excerpt and break
    EXCERPT=$(echo "$BODY" | tr -d '\n' | sed 's/"/'"'"'/g' | awk '{print substr($0,1,400)}')
    break
  done

  # CSV safe line
  ts=$(date -Iseconds)
  okstr=$([ "$OK" = true ] && echo "yes" || echo "no")
  printf '%s,%s,%s,%s,%s,"%s"\n' "$ts" "$model" "$HTTP_STATUS" "$TIME_MS" "$okstr" "$EXCERPT" >> "$OUT"

  echo " -> status: $HTTP_STATUS time_ms: ${TIME_MS} ok: $okstr"
done

echo
echo "Wrote results to: $OUT"
echo
cat <<'EXPL'
Interpretation:
  200 (2xx) -> request accepted; model returned output (OK).
  401 -> No auth credentials / invalid key. Check OPENROUTER_API_KEY value.
  403 -> Forbidden / permissions problem (account can't access model).
  404 -> Model not found (ID mismatch).
  429 -> Rate-limited. Retry later or reduce requests.
  500/502/503 -> provider/server error; retry later.

Next steps:
 - If you see 401 for every model: your key is invalid or not present (re-export and re-run).
 - If 401 only for some models: key exists but account lacks permission for those models.
 - If 404: use exact model ID shown in OpenRouter UI.
 - If 429: wait and rerun or slow down the checks.
EXPL
