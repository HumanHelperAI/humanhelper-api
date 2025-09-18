#!/data/data/com.termux/files/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

python -V || true
[ ! -d .venv ] && python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

mkdir -p backups logs

python - <<'PY'
from database import init_db
init_db()
print("DB schema ✅ ready")
PY

echo "Setup complete ✅"
echo "Next: ./start.sh"
