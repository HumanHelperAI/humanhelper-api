#!/data/data/com.termux/files/usr/bin/env bash
set -e
SESSION="humanhelper"
LOGFILE="logs/app.log"

[ -f .env ] && set -o allexport && source .env && set +o allexport
FLASK_HOST="${FLASK_HOST:-0.0.0.0}"
FLASK_PORT="${FLASK_PORT:-5000}"

source .venv/bin/activate

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION"
fi

CMD="python app.py"
tmux kill-window -t "$SESSION:api" 2>/dev/null || true
tmux new-window -t "$SESSION" -n api "mkdir -p logs; echo 'Starting API…' > $LOGFILE; $CMD >> $LOGFILE 2>&1"

echo "API started in tmux session '$SESSION', window 'api'."
echo "View logs: tail -f $LOGFILE"
echo "Attach:    tmux attach -t $SESSION"
