#!/usr/bin/env bash
# test_models.sh - improved: writes to ./model_results, iterates model list, tests OpenRouter/OpenAI/DeepSeek/LLaMA
# Save as test_models.sh, then:
#   chmod +x test_models.sh
#   ./test_models.sh
set -euo pipefail

# --- Config (read from env if set) ---
OPENROUTER_URL="${OPENROUTER_URL:-https://openrouter.ai/v1/chat/completions}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
DEEPSEEK_URL="${DEEPSEEK_URL:-https://api.deepseek.com/v1/chat/completions}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
LLAMA_URL="${LLAMA_SERVER_URL:-http://127.0.0.1:5001/completion}"

# Directory for results (local, safe)
RESULT_DIR="./model_results"
mkdir -p "$RESULT_DIR"

echo "Results will be written to $RESULT_DIR"
echo

# The list below is the prioritized free models you mentioned.
# Format for OpenRouter: model names typically like "gpt-oss-20b" or "openrouter/xxx"
FREE_MODELS=(
  "gpt-oss-20b"
  "gpt-oss-120b"
  "deepseek-v3.1-free"
  "deepseek-r1-free"
  "deepseek-r1-0528-qwen3-8b"   # example token; adjust if your provider expects different name
  "nous-deephermes-3-llama-3-8b-preview"
  "google-gemini-2.0-flash-exp"
)

# helper to run a curl POST and capture outputs
run_and_capture() {
  name="$1"   # short name used for file naming
  shift
  outf="$RESULT_DIR/$name.out"
  errf="$RESULT_DIR/$name.err"
  rm -f "$outf" "$errf"
  echo "---- running: $name ----"
  if curl -sS "$@" > "$outf" 2> "$errf"; then
    echo "OK: $name -> $outf"
  else
    rc=$?
    echo "FAILED: $name (exit $rc) -> see $errf / $outf"
  fi
}

# quick DNS / reachability checks
echo "=== DNS / Reachability ==="
echo "OpenRouter URL: $OPENROUTER_URL"
echo "OpenAI configured: ${OPENAI_API_KEY:+yes}"
echo "DeepSeek configured: ${DEEPSEEK_API_KEY:+yes}"
echo "LLaMA URL: $LLAMA_URL"
echo
echo "dig openrouter.ai A:"
dig +short openrouter.ai || true
echo "dig api.openrouter.ai A:"
dig +short api.openrouter.ai || true
echo

# 1) Test OpenRouter across FREE_MODELS (if API key present)
if [ -n "$OPENROUTER_API_KEY" ]; then
  for mdl in "${FREE_MODELS[@]}"; do
    safe_name="openrouter__${mdl//[^a-zA-Z0-9._-]/_}"
    # Try main OPENROUTER_URL. Some networks may need the 'openrouter.ai' host (fallback handled by app).
    run_and_capture "$safe_name" \
      -X POST "$OPENROUTER_URL" \
      -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      -H "Content-Type: application/json" \
      --data-raw "{\"model\":\"$mdl\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one line\"}],\"max_tokens\":60}"
    sleep 0.4
  done
else
  echo "Skipping OpenRouter tests: OPENROUTER_API_KEY empty"
fi

# 2) Test OpenAI (standard Chat Completions) if key present
if [ -n "$OPENAI_API_KEY" ]; then
  run_and_capture "openai_gpt35" \
    -X POST "https://api.openai.com/v1/chat/completions" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    --data-raw '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Say hi in one line"}],"max_tokens":60}'
else
  echo "Skipping OpenAI test: OPENAI_API_KEY empty"
fi

# 3) Test DeepSeek (if configured)
if [ -n "$DEEPSEEK_API_KEY" ]; then
  run_and_capture "deepseek_v3.1_free" \
    -X POST "$DEEPSEEK_URL" \
    -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
    -H "Content-Type: application/json" \
    --data-raw '{"model":"deepseek-v3.1-free","messages":[{"role":"user","content":"Say hi in one line"}],"max_tokens":60}'
else
  echo "Skipping DeepSeek test: DEEPSEEK_API_KEY empty"
fi

# 4) Test local LLaMA server (if present)
run_and_capture "llama_local" \
  -X POST "$LLAMA_URL" \
  -H "Content-Type: application/json" \
  --data-raw '{"prompt":"Say hi in one line","n_predict":128}'

echo
echo "=== SUMMARY ==="
ls -1 "$RESULT_DIR" || true
echo
echo "To inspect a result, run e.g.:"
echo "  sed -n '1,200p' $RESULT_DIR/openrouter__gpt-oss-20b.out"
echo "Or open the .err to see curl errors."
echo
echo "Done."
