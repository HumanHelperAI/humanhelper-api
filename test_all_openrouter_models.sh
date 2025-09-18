#!/usr/bin/env bash
set -euo pipefail

: "${OPENROUTER_API_KEY:?Please export OPENROUTER_API_KEY=sk-...}"
MODELS="${FREE_MODELS_PRIORITY:-"deepseek/deepseek-v3.1-free deepseek/r1-free deepseek/qwen3-8b-free nous/deephermes-3-llama-3-8b openai/gpt-oss-20b openai/gpt-oss-120b google/gemini-2.0-flash-exp"}"
OUT="./openrouter_model_check_results.csv"
echo "timestamp,model,http_status,time_ms,ok,excerpt" > "$OUT"

for model in $MODELS; do
  ts="$(date --iso-8601=seconds)"
  echo "[$ts] Testing model: $model"
  payload=$(jq -n --arg m "$model" --arg p "HumanHelper test" --argjson mt 64 '{model:$m, messages:[{role:"user", content:$p}], max_tokens:$mt}')
  t0=$(date +%s%3N)
  http=$(curl -s -w "%{http_code}" -o /tmp/openrouter_resp.json \
    -X POST "https://openrouter.ai/api/v1/chat/completions" \
    -H "Authorization: Bearer ${OPENROUTER_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$payload" --max-time 20) || http="000"
  t1=$(date +%s%3N); dt=$((t1-t0))
  excerpt=$(jq -r '(.choices[0].message.content // .error.message // .choices[0].message.reasoning // .choices[0].message.reasoning_details[0].text) // empty' /tmp/openrouter_resp.json 2>/dev/null | tr '\n' ' ' | sed 's/"/""/g' | cut -c1-200 || true)
  ok="no"; [ "$http" -ge 200 ] && [ "$http" -lt 300 ] && ok="yes"
  echo "$ts,$model,$http,$dt,$ok,\"$excerpt\"" >> "$OUT"
  echo " -> status: $http time_ms:$dt ok:$ok excerpt: ${excerpt:0:80}"
done

echo "Wrote $OUT"
