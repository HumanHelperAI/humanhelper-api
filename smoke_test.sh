#!/usr/bin/env bash
set -euo pipefail
BASE="http://127.0.0.1:5000"
LLAMA="http://127.0.0.1:5001/completion"
ADMIN_TOKEN="${ADMIN_TOKEN:-Muralidhar}"

echo "=== HumanHelper smoke tests ==="
echo

# Helper
call() {
  local method="$1"; shift
  curl -s -X "$method" "$@" || echo "curl failed"
}

# 0) service health
echo -n "Health: "
health=$(call GET "$BASE/health")
if echo "$health" | grep -q '"status": "ok"'; then echo "PASS"; else echo "FAIL: $health"; fi

# 1) signup (random mobile to avoid conflict)
MOBILE="999${RANDOM:0:4}"
echo "Testing signup with mobile: $MOBILE"
signup=$(curl -s -X POST "$BASE/signup" -H "Content-Type: application/json" -d "{\"name\":\"TT\",\"mobile\":\"$MOBILE\",\"aadhar\":\"1111\",\"password\":\"pass123\"}")
echo " signup -> $signup"
if echo "$signup" | grep -q "User registered"; then echo "SIGNUP PASS"; else echo "SIGNUP FAIL"; fi
echo

# 2) login
echo "Testing login"
login=$(curl -s -X POST "$BASE/login" -H "Content-Type: application/json" -d "{\"mobile\":\"$MOBILE\",\"password\":\"pass123\"}")
echo " login -> $login"
if echo "$login" | grep -q "Login successful"; then echo "LOGIN PASS"; else echo "LOGIN FAIL"; fi
echo

# 3) deposit (simulate)
echo "Testing deposit (adds 10)"
dep=$(curl -s -X POST "$BASE/deposit" -H "Content-Type: application/json" -d "{\"mobile\":\"$MOBILE\",\"amount\":10}")
echo " deposit -> $dep"
if echo "$dep" | grep -q "message"; then echo "DEPOSIT CALL OK"; else echo "DEPOSIT FAIL"; fi
echo

# 4) check balance
bal=$(curl -s "$BASE/balance/$MOBILE")
echo " balance -> $bal"
echo

# 5) earn endpoint (call reward_user)
echo "Testing earn endpoint (duration 10)"
earn=$(curl -s -X POST "$BASE/earn" -H "Content-Type: application/json" -d "{\"mobile\":\"$MOBILE\",\"video_id\":\"v1\",\"content_type\":\"video\",\"duration\":10}")
echo " earn -> $earn"
echo

# 6) charity balance
char=$(curl -s "$BASE/charity/balance")
echo " charity balance -> $char"
echo

# 7) Admin list users (requires admin header)
echo "Admin list users (should require ADMIN_TOKEN)"
adm=$(curl -s -X GET "$BASE/admin/users" -H "X-Admin-Token: $ADMIN_TOKEN")
echo " admin/users -> $adm"
echo

# 8) Test AI LLaMA availability direct
echo -n "LLAMA direct call: "
ll=$(curl -s -X POST "$LLAMA" -H "Content-Type: application/json" -d '{"prompt":"Hello from smoke test","n_predict":32}')
if echo "$ll" | grep -q '"content"'; then echo "PASS"; else echo "FAIL: $ll"; fi
echo

# 9) Test /ai/humanhelper endpoint (non-premium)
echo -n "AI humanhelper (no premium): "
ai=$(curl -s -X POST "$BASE/ai/humanhelper" -H "Content-Type: application/json" -d '{"prompt":"Explain how to boil an egg"}')
echo "$ai" | sed -n '1,6p'
if echo "$ai" | grep -q '"provider"'; then echo "AI ENDPOINT OK"; else echo "AI ENDPOINT FAIL"; fi
echo

# 10) Test premium flow (admin token = premium)
echo -n "AI humanhelper (premium via X-Admin-Token): "
ai2=$(curl -s -X POST "$BASE/ai/humanhelper" -H "Content-Type: application/json" -H "X-Admin-Token: $ADMIN_TOKEN" -d '{"prompt":"Write a professional email asking for a quote"}')
echo "$ai2" | sed -n '1,6p'
if echo "$ai2" | grep -q '"provider"'; then echo "AI PREMIUM PATH OK"; else echo "AI PREMIUM PATH FAIL"; fi
echo

# 11) Test ai/ask simpler endpoint
echo -n "AI /ai/ask: "
a3=$(curl -s -X POST "$BASE/ai/ask" -H "Content-Type: application/json" -d '{"prompt":"Say hi"}')
echo "$a3" | sed -n '1,6p'
echo

echo "=== smoke tests finished ==="
