#!/usr/bin/env bash
set -euo pipefail

# e2e_ai_check.sh - run full end-to-end checks against local API.
# Usage:
#   cd ~/humanhelper/backend
#   ./e2e_ai_check.sh

BASE_URL="${BASE_URL:-http://127.0.0.1:5000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
# helper: pretty json if jq available
pretty() {
  if command -v jq >/dev/null 2>&1; then
    jq .
  else
    python -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))"
  fi
}

echo "========== 0) Health =========="
curl -s "${BASE_URL}/health" | pretty
echo

# make random suffixes so we don't hit "Already watched"
RAND=$(python - <<'PY'
import uuid, time
print(uuid.uuid4().hex[:8])
PY
)
MOBILE="99${R:=${RAND:0:6}}"  # not used if API expects a format; we keep earlier one below

# We'll use a reproducible mobile for easier later admin tests:
MOBILE="9912345678"
NAME="Tester"
AADHAR="$(date +%s | tail -c8)"
PAN="PAN${RAND^^}X"
PASSWORD="pw"

echo "========== 1) Signup =========="
signup_payload=$(jq -n \
  --arg name "$NAME" \
  --arg mobile "$MOBILE" \
  --arg aadhar "$AADHAR" \
  --arg pan "$PAN" \
  --arg password "$PASSWORD" \
  '{name:$name,mobile:$mobile,aadhar:$aadhar,pan:$pan,password:$password}')
curl -s -X POST "${BASE_URL}/signup" -H "Content-Type: application/json" -d "$signup_payload" | pretty
echo

echo "========== 2) Login =========="
login_payload=$(jq -n --arg mobile "$MOBILE" --arg password "$PASSWORD" '{mobile:$mobile,password:$password}')
curl -s -X POST "${BASE_URL}/login" -H "Content-Type: application/json" -d "$login_payload" | pretty
echo

echo "========== 3) Deposit ₹50 =========="
deposit_payload=$(jq -n --arg mobile "$MOBILE" --argjson amount 50 '{mobile:$mobile,amount:$amount}')
curl -s -X POST "${BASE_URL}/deposit" -H "Content-Type: application/json" -d "$deposit_payload" | pretty
echo

echo "========== 4) Withdraw ₹20 =========="
withdraw_payload=$(jq -n --arg mobile "$MOBILE" --argjson amount 20 '{mobile:$mobile,amount:$amount}')
curl -s -X POST "${BASE_URL}/withdraw" -H "Content-Type: application/json" -d "$withdraw_payload" | pretty
echo

# generate new random IDs for earn checks so we don't hit duplicate watch protection
VID=$(python - <<'PY'
import uuid
print("vid-"+uuid.uuid4().hex[:10])
PY
)
SHORTID=$(python - <<'PY'
import uuid
print("short-"+uuid.uuid4().hex[:8])
PY
)

echo "========== 5) Earn (video 120s) =========="
earn_payload=$(jq -n --arg mobile "$MOBILE" --arg video_id "$VID" --arg content_type "video" --argjson duration 120 '{mobile:$mobile,video_id:$video_id,content_type:$content_type,duration:$duration}')
curl -s -X POST "${BASE_URL}/earn" -H "Content-Type: application/json" -d "$earn_payload" | pretty
echo

echo "========== 6) Earn (short 20s) =========="
earn_payload2=$(jq -n --arg mobile "$MOBILE" --arg video_id "$SHORTID" --arg content_type "short" --argjson duration 20 '{mobile:$mobile,video_id:$video_id,content_type:$content_type,duration:$duration}')
curl -s -X POST "${BASE_URL}/earn" -H "Content-Type: application/json" -d "$earn_payload2" | pretty
echo

echo "========== 7) Balance =========="
curl -s "${BASE_URL}/balance/${MOBILE}" | pretty
echo

echo "========== 8) Transactions =========="
curl -s "${BASE_URL}/transactions/${MOBILE}" | pretty
echo

echo "========== 9) Charity balance =========="
curl -s "${BASE_URL}/charity/balance" | pretty
echo

echo "========== 10) Admin: List users =========="
curl -s -X GET "${BASE_URL}/admin/users" -H "X-Admin-Token: ${ADMIN_TOKEN}" | pretty
echo

echo "========== 11) Admin: Adjust balance +10 =========="
adjust_payload=$(jq -n --arg mobile "$MOBILE" --argjson delta 10 --arg note "bonus" '{mobile:$mobile,delta:$delta,note:$note}')
curl -s -X POST "${BASE_URL}/admin/balance/adjust" -H "Content-Type: application/json" -H "X-Admin-Token: ${ADMIN_TOKEN}" -d "$adjust_payload" | pretty
echo

echo "========== 12) Admin: Delete user =========="
del_payload=$(jq -n --arg mobile "$MOBILE" '{mobile:$mobile}')
curl -s -X POST "${BASE_URL}/admin/user/delete" -H "Content-Type: application/json" -H "X-Admin-Token: ${ADMIN_TOKEN}" -d "$del_payload" | pretty
echo

# AI checks - only test; they will report "not configured" if keys are missing
echo "========== 13) AI Test: OpenAI =========="
ai_openai_payload=$(jq -n --arg provider "openai" --arg prompt "Say hi from HumanHelper e2e test" '{provider:$provider,prompt:$prompt}')
curl -s -X POST "${BASE_URL}/ai/ask" -H "Content-Type: application/json" -d "$ai_openai_payload" | pretty
echo

echo "========== 14) AI Test: Gemini =========="
ai_gemini_payload=$(jq -n --arg provider "gemini" --arg prompt "Say hi from HumanHelper e2e test" '{provider:$provider,prompt:$prompt}')
curl -s -X POST "${BASE_URL}/ai/ask" -H "Content-Type: application/json" -d "$ai_gemini_payload" | pretty
echo

echo "========== 15) AI Test: DeepSeek =========="
ai_deep_payload=$(jq -n --arg provider "deepseek" --arg prompt "Search for 'term' in sample documents'" '{provider:$provider,prompt:$prompt}')
curl -s -X POST "${BASE_URL}/ai/ask" -H "Content-Type: application/json" -d "$ai_deep_payload" | pretty
echo

echo "========== ✅ DONE =========="
