#!/usr/bin/env bash
set -eu
BASE="${BASE:-https://api.humanhelperai.in}"
MOBILE="${MOBILE:-9876543212}"
PASS="${PASS:-Passw0rd!}"
NAME="${NAME:-Smoke Tester}"
EMAIL="${EMAIL:-smoke@test.com}"
ADDR="${ADDR:-Hyderabad}"

say() { printf "\n==> %s\n" "$*"; }

say "health"; curl -s "$BASE/health" | jq -r '.status'

say "register"; curl -s -X POST "$BASE/auth/register" -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg n "$NAME" --arg m "$MOBILE" --arg p "$PASS" --arg e "$EMAIL" --arg a "$ADDR" \
       '{full_name:$n,mobile:$m,password:$p,email:$e,address:$a}')" | jq .

echo "Enter OTP from logs:"; read -r OTP
say "verify"; curl -s -X POST "$BASE/auth/verify" -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg m "$MOBILE" --arg c "$OTP" '{mobile:$m,code:$c}')" | jq .

say "login"
LOGIN=$(curl -s -X POST "$BASE/auth/login" -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg m "$MOBILE" --arg p "$PASS" '{mobile:$m,password:$p}')" )
echo "$LOGIN" | jq .
ACCESS=$(echo "$LOGIN"  | jq -r '.access')
REFRESH=$(echo "$LOGIN" | jq -r '.refresh')

say "whoami"; curl -s "$BASE/whoami" -H "Authorization: Bearer $ACCESS" | jq .

say "ai/ask"; curl -s -X POST "$BASE/ai/ask" -H 'Content-Type: application/json' \
  -d '{"prompt":"Quick hello!"}' | jq -r '.provider,.answer' | sed -n '1p;2p'
