#!/usr/bin/env bash
set -euo pipefail

# --- EDIT THESE 2 IF NEEDED ---
REMOTE=origin
BRANCH=main
# ------------------------------

echo "🔧 Lint: syntax check"
python -m py_compile app.py || { echo "❌ Python syntax error"; exit 1; }

echo "📦 Git add/commit/push"
git add -A
git commit -m "redeploy: migrations + wallet fixes" || echo "ℹ️ nothing to commit"
git push "$REMOTE" "$BRANCH"

echo "⏳ Waiting 25s for Railway to build/deploy…"
sleep 25

# --- Runtime smoke tests ---
export BASE="${BASE:-https://api.humanhelperai.in}"

echo "🩺 /health"
curl -fsS "$BASE/health" | jq .

echo "🏷️  /version"
curl -fsS "$BASE/version" | jq .

# Admin quick checks (optional)
if [[ -n "${ADMIN_TOKEN:-}" ]]; then
  echo "💼 /admin/wallet/fees"
  curl -fsS "$BASE/admin/wallet/fees" -H "X-Admin-Token: $ADMIN_TOKEN" | jq .
fi

echo "✅ Redeploy smoke tests done."
