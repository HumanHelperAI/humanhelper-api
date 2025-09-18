#!/usr/bin/env bash
SESSION=backup_watcher
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "Stopped watcher session $SESSION"
else
  echo "Watcher not running"
fi
