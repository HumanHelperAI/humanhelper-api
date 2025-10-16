#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-https://api.humanhelperai.in}"   # change if needed
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
PASS="REQUIRE_ENV_PASS"
A_MOBILE="9876509001"
B_MOBILE="9876509002"

echo "BASE=$BASE"
echo "Check version"
curl -s "$BASE/version" | jq .

echo; echo "Check health + DB engine"
curl -s "$BASE/health" | jq . || true
curl -s "$BASE/debug/whichdb" | jq . || true

echo; echo "Register A ($A_MOBILE)"
curl -s -X POST "$BASE/auth/register" -H 'Content-Type: application/json' \
  -d "{\"full_name\":\"Smoke A\",\"mobile\":\"$A_MOBILE\",\"password\":\"$PASS\",\"address\":\"addr\"}" | jq .

echo; echo "Register B ($B_MOBILE)"
curl -s -X POST "$BASE/auth/register" -H 'Content-Type: application/json' \
  -d "{\"full_name\":\"Smoke B\",\"mobile\":\"$B_MOBILE\",\"password\":\"$PASS\",\"address\":\"addr\"}" | jq .

echo; echo "Simulate verify for A/B — if OTP flow used, paste OTP here"
# If your app prints OTP to logs (SEND_VERIFICATION_MODE=console), read from logs and set these:
OTP_A="${OTP_A:-123456}"
OTP_B="${OTP_B:-123456}"

curl -s -X POST "$BASE/auth/verify" -H 'Content-Type: application/json' \
  -d "{\"mobile\":\"$A_MOBILE\",\"code\":\"$OTP_A\"}" | jq .
curl -s -X POST "$BASE/auth/verify" -H 'Content-Type: application/json' \
  -d "{\"mobile\":\"$B_MOBILE\",\"code\":\"$OTP_B\"}" | jq .

echo; echo "Login A and B"
LOGIN_A=$(curl -s -X POST "$BASE/auth/login" -H 'Content-Type: application/json' \
  -d "{\"mobile\":\"$A_MOBILE\",\"password\":\"$PASS\"}")
echo "$LOGIN_A" | jq .
ACCESS_A=$(echo "$LOGIN_A" | jq -r .access // empty)
REFRESH_A=$(echo "$LOGIN_A" | jq -r .refresh // empty)

LOGIN_B=$(curl -s -X POST "$BASE/auth/login" -H 'Content-Type: application/json' \
  -d "{\"mobile\":\"$B_MOBILE\",\"password\":\"$PASS\"}")
echo "$LOGIN_B" | jq .
ACCESS_B=$(echo "$LOGIN_B" | jq -r .access // empty)
REFRESH_B=$(echo "$LOGIN_B" | jq -r .refresh // empty)

echo; echo "Whoami A"
curl -s "$BASE/whoami" -H "Authorization: Bearer $ACCESS_A" | jq . || true

echo; echo "Admin: seed A with 500 (uses ADMIN_TOKEN)"
# Need UID_A — fetch via whoami or admin users
WHO_A=$(curl -s "$BASE/whoami" -H "Authorization: Bearer $ACCESS_A" || echo "{}")
UID_A=$(echo "$WHO_A" | jq -r .user_id // empty)
if [ -z "$UID_A" ] || [ "$UID_A" = "null" ]; then
  echo "Cannot get UID_A from whoami; falling back to admin /users list"
  curl -s -H "X-Admin-Token: $ADMIN_TOKEN" "$BASE/admin/users?limit=5" | jq .
  # If you find the id manually, set UID_A accordingly
fi

if [ -n "$UID_A" ] && [ "$UID_A" != "null" ]; then
  curl -s -X POST "$BASE/admin/user/adjust" \
    -H "X-Admin-Token: $ADMIN_TOKEN" -H 'Content-Type: application/json' \
    -d "{\"user_id\": $UID_A, \"delta\": 500, \"note\": \"seed\"}" | jq .
fi

echo; echo "Transfer A -> B: 25"
# Get UID_B
WHO_B=$(curl -s "$BASE/whoami" -H "Authorization: Bearer $ACCESS_B" || echo "{}")
UID_B=$(echo "$WHO_B" | jq -r .user_id // empty)
if [ -z "$UID_B" ] || [ "$UID_B" = "null" ]; then
  echo "Cannot get UID_B from whoami; use admin/users to find id"
fi
curl -s -X POST "$BASE/wallet/transfer" \
  -H "Authorization: Bearer $ACCESS_A" -H 'Content-Type: application/json' \
  -d "{\"receiver_id\": $UID_B, \"amount\": 25}" | jq .

echo; echo "Balances A and B"
curl -s "$BASE/wallet/balance" -H "Authorization: Bearer $ACCESS_A" | jq .
curl -s "$BASE/wallet/balance" -H "Authorization: Bearer $ACCESS_B" | jq .

echo; echo "Create withdraw from A (small amount) — requires Razorpay config to succeed"
curl -s -X POST "$BASE/wallet/withdraw" -H "Authorization: Bearer $ACCESS_A" -H 'Content-Type: application/json' \
  -d '{"amount": 100, "upi_id":"test@upi"}' | jq .

echo; echo "Admin: fee pool"
curl -s "$BASE/admin/wallet/fees" -H "X-Admin-Token: $ADMIN_TOKEN" | jq .

echo; echo "Admin: repair-withdrawals (manual)"
curl -s -X POST "$BASE/admin/tools/repair-withdrawals" -H "X-Admin-Token: $ADMIN_TOKEN" | jq .

echo; echo "Transactions A"
curl -s "$BASE/wallet/transactions?limit=20" -H "Authorization: Bearer $ACCESS_A" | jq .

echo; echo "Done smoke tests"
