#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:5000}"
MOBILE="${MOBILE:-9111111111}"
PAN="${PAN:-TESTPAN1X}"
AADHAR="${AADHAR:-123412341234}"
PASS="${PASS:-pw}"
NAME="${NAME:-TestUser}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"   # set this in env if you want admin delete to run

step () { echo -e "\n========== $1 =========="; }

req () {
  # usage: req METHOD PATH JSON
  local method="$1"; shift
  local path="$1"; shift
  local data="${1:-}"
  if [[ -n "$data" ]]; then
    curl -sS -f -X "$method" "$API_BASE$path" -H "Content-Type: application/json" -d "$data" || {
      echo "[ERROR] $method $path failed"; exit 1; }
  else
    curl -sS -f -X "$method" "$API_BASE$path" || { echo "[ERROR] $method $path failed"; exit 1; }
  fi
  echo
}

step "0) Health"
req GET /health

step "1) Signup"
req POST /signup \
  "{\"name\":\"$NAME\",\"mobile\":\"$MOBILE\",\"aadhar\":\"$AADHAR\",\"pan\":\"$PAN\",\"password\":\"$PASS\"}"

step "2) Login"
req POST /login \
  "{\"mobile\":\"$MOBILE\",\"password\":\"$PASS\"}"

step "3) Deposit ₹100"
req POST /deposit \
  "{\"mobile\":\"$MOBILE\",\"amount\":100}"

step "4) Withdraw ₹50 (7% charity)"
req POST /withdraw \
  "{\"mobile\":\"$MOBILE\",\"amount\":50}"

step "5) Earn (video 120s → user +₹0.24, charity ₹0.06)"
req POST /earn \
  "{\"mobile\":\"$MOBILE\",\"video_id\":\"vid_e2e_120\",\"content_type\":\"video\",\"duration\":120}"

step "6) Earn (short 20s → user +₹0.02, charity ₹0.01)"
req POST /earn \
  "{\"mobile\":\"$MOBILE\",\"video_id\":\"short_e2e_20\",\"content_type\":\"short\",\"duration\":20}"

step "7) User balance"
req GET "/balance/$MOBILE"

step "8) User transactions (latest first)"
req GET "/transactions/$MOBILE"

step "9) Charity balance"
req GET /charity/balance

step "10) Concurrency burst (5 quick deposits of ₹1)"
for i in 1 2 3 4 5; do
  req POST /deposit "{\"mobile\":\"$MOBILE\",\"amount\":1}" >/dev/null
done
echo "Done."

step "11) Balance after burst (+₹5 expected)"
req GET "/balance/$MOBILE"

if [[ -n "$ADMIN_TOKEN" ]]; then
  step "12) ADMIN delete user by PAN (queued via writer)"
  curl -sS -f -X POST "$API_BASE/admin/delete_user" \
    -H "Content-Type: application/json" \
    -H "X-Admin-Token: $ADMIN_TOKEN" \
    -d "{\"pan\":\"$PAN\"}" || { echo "[ERROR] admin delete failed"; exit 1; }
  echo

  step "13) Try login after delete (should fail)"
  set +e
  curl -sS -X POST "$API_BASE/login" -H "Content-Type: application/json" \
    -d "{\"mobile\":\"$MOBILE\",\"password\":\"$PASS\"}"
  echo
  set -e
else
  step "12) ADMIN delete user (skipped, no ADMIN_TOKEN in env)"
fi

echo -e "\n========== ✅ DONE =========="
echo "Mobile: $MOBILE | PAN: $PAN | Aadhar: $AADHAR"
