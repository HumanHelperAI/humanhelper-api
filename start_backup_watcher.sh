#!/usr/bin/env bash
# start_backup_watcher.sh
SESSION=backup_watcher
cd "$(dirname "$0")"

# start in tmux so it runs detached (tmux must be installed)
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Watcher already running (tmux session: $SESSION)"
  tmux attach -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash -lc 'export HOME=\"$HOME\"; exec ./pendrive_watcher.sh'"

echo "Watcher started in tmux session '$SESSION'. View logs with: tmux attach -t $SESSION"
