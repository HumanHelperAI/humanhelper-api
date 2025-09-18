#!/usr/bin/env bash
# run_services.sh
# Robust service manager for HumanHelper: start/stop/status/attach/refresh
# Uses env vars HH_* if set, otherwise sensible defaults.

set -eu

# ---- Configurable via environment (or fallback defaults) ----
HH_PROJECT_DIR="${HH_PROJECT_DIR:-$HOME/humanhelper/backend}"
HH_VENV_ACTIVATE="${HH_VENV_ACTIVATE:-$HH_PROJECT_DIR/.venv/bin/activate}"
HH_LLAMA_BIN="${HH_LLAMA_BIN:-$HH_PROJECT_DIR/llama.cpp/build/bin/llama-server}"
HH_LLAMA_MODEL="${HH_LLAMA_MODEL:-$HOME/models/tinyllama-q4.gguf}"
HH_LLAMA_PORT="${HH_LLAMA_PORT:-5001}"
HH_FLASK_PORT="${HH_FLASK_PORT:-5000}"
HH_FLASK_APP="${HH_FLASK_APP:-$HH_PROJECT_DIR/app.py}"
HH_LOG_DIR="${HH_LOG_DIR:-$HH_PROJECT_DIR}"
HH_LLAMA_LOG="${HH_LLAMA_LOG:-$HH_LOG_DIR/llama_stdout.log}"
HH_FLASK_LOG="${HH_FLASK_LOG:-$HH_LOG_DIR/flask_stdout.log}"
TMUX_SESSION="${TMUX_SESSION:-humanhelper}"

mkdir -p "$HH_LOG_DIR" 2>/dev/null || true

# ---- helpers ----
_pids_for() {
  # returns newline-separated pids for processes whose command line contains $1
  ps aux | grep -F "$1" | grep -v grep | awk '{print $2}' || true
}

_is_running() {
  local pattern="$1"
  local pids
  pids=$(_pids_for "$pattern")
  if [ -n "$pids" ]; then
    echo "$pids"
    return 0
  fi
  return 1
}

# ---- LLaMA ----
start_llama_nohup() {
  if [ ! -x "$HH_LLAMA_BIN" ]; then
    echo "ERROR: llama binary not executable: $HH_LLAMA_BIN"
    return 2
  fi
  if _is_running "$HH_LLAMA_BIN"; then
    echo "LLaMA already running (pids):"
    _is_running "$HH_LLAMA_BIN"
    return 0
  fi
  echo "Starting LLaMA on port $HH_LLAMA_PORT ..."
  nohup "$HH_LLAMA_BIN" -m "$HH_LLAMA_MODEL" --port "$HH_LLAMA_PORT" --threads 4 -c 2048 > "$HH_LLAMA_LOG" 2>&1 &
  sleep 0.5
  _is_running "$HH_LLAMA_BIN" || echo "(LLaMA start may have failed - check $HH_LLAMA_LOG)"
}

start_llama_tmux() {
  # create tmux session/window if necessary
  if ! command -v tmux >/dev/null 2>&1; then
    start_llama_nohup
    return
  fi

  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    # create new window for llama if doesn't exist
    if ! tmux list-windows -t "$TMUX_SESSION" 2>/dev/null | grep -q '^llama'; then
      tmux new-window -t "$TMUX_SESSION" -n llama "bash -lc 'exec \"$HH_LLAMA_BIN\" -m \"$HH_LLAMA_MODEL\" --port \"$HH_LLAMA_PORT\" --threads 4 -c 2048 2>&1 | tee \"$HH_LLAMA_LOG\"'"
    else
      echo "tmux session exists and llama window already present."
    fi
  else
    tmux new-session -d -s "$TMUX_SESSION" -n llama "bash -lc 'exec \"$HH_LLAMA_BIN\" -m \"$HH_LLAMA_MODEL\" --port \"$HH_LLAMA_PORT\" --threads 4 -c 2048 2>&1 | tee \"$HH_LLAMA_LOG\"'"
  fi
  sleep 0.4
  _is_running "$HH_LLAMA_BIN" || echo "(LLaMA start may have failed - check $HH_LLAMA_LOG)"
}

stop_llama() {
  local pids
  pids=$(_pids_for "$HH_LLAMA_BIN")
  if [ -z "$pids" ]; then
    echo "No running LLaMA process found."
    return 0
  fi
  echo "Stopping LLaMA pid(s):"
  echo "$pids"
  echo "$pids" | xargs -r kill
  sleep 0.5
  local still
  still=$(_pids_for "$HH_LLAMA_BIN")
  if [ -n "$still" ]; then
    echo "Forcing kill for: $still"
    echo "$still" | xargs -r kill -9
  fi
  # also cleanup tmux window if session exists
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux kill-window -t "$TMUX_SESSION":llama 2>/dev/null || true
  fi
  echo "LLaMA stopped."
}

# ---- Flask ----
start_flask_nohup() {
  if [ ! -f "$HH_FLASK_APP" ]; then
    echo "ERROR: Flask app not found: $HH_FLASK_APP"
    return 2
  fi
  if _is_running "python $HH_FLASK_APP"; then
    echo "Flask already running (pids):"
    _is_running "python $HH_FLASK_APP"
    return 0
  fi

  if [ -f "$HH_VENV_ACTIVATE" ]; then
    echo "Activating venv: $HH_VENV_ACTIVATE"
    nohup bash -lc "source \"$HH_VENV_ACTIVATE\" && python \"$HH_FLASK_APP\"" > "$HH_FLASK_LOG" 2>&1 &
  else
    echo "Starting Flask without venv"
    nohup python "$HH_FLASK_APP" > "$HH_FLASK_LOG" 2>&1 &
  fi
  sleep 0.6
  _is_running "python $HH_FLASK_APP" || echo "(Flask start may have failed - check $HH_FLASK_LOG)"
}

start_flask_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    start_flask_nohup
    return
  fi

  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    if ! tmux list-windows -t "$TMUX_SESSION" 2>/dev/null | grep -q '^flask'; then
      tmux new-window -t "$TMUX_SESSION" -n flask "bash -lc 'source \"$HH_VENV_ACTIVATE\" 2>/dev/null || true; exec python \"$HH_FLASK_APP\" 2>&1 | tee \"$HH_FLASK_LOG\"'"
    else
      echo "tmux session exists and flask window already present."
    fi
  else
    tmux new-session -d -s "$TMUX_SESSION" -n flask "bash -lc 'source \"$HH_VENV_ACTIVATE\" 2>/dev/null || true; exec python \"$HH_FLASK_APP\" 2>&1 | tee \"$HH_FLASK_LOG\"'"
  fi
  sleep 0.4
  _is_running "python $HH_FLASK_APP" || echo "(Flask start may have failed - check $HH_FLASK_LOG)"
}

stop_flask() {
  local pids
  pids=$(_pids_for "python $HH_FLASK_APP")
  if [ -z "$pids" ]; then
    echo "No running Flask process found."
    return 0
  fi
  echo "Stopping Flask pid(s):"
  echo "$pids"
  echo "$pids" | xargs -r kill
  sleep 0.5
  local still
  still=$(_pids_for "python $HH_FLASK_APP")
  if [ -n "$still" ]; then
    echo "Forcing kill for: $still"
    echo "$still" | xargs -r kill -9
  fi
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux kill-window -t "$TMUX_SESSION":flask 2>/dev/null || true
  fi
  echo "Flask stopped."
}

# ---- high level ----
_status() {
  echo "---- Status ----"
  echo "LLaMA ($HH_LLAMA_BIN) on port $HH_LLAMA_PORT:"
  _is_running "$HH_LLAMA_BIN" || echo "(not running)"
  echo
  echo "Flask ($HH_FLASK_APP) on port $HH_FLASK_PORT:"
  _is_running "python $HH_FLASK_APP" || echo "(not running)"
  echo
  echo "Logs:"
  echo " LLaMA -> $HH_LLAMA_LOG"
  echo " Flask -> $HH_FLASK_LOG"
}

_attach() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found. No attach available; check logs instead (tail -f $HH_LLAMA_LOG / $HH_FLASK_LOG)."
    return 2
  fi
  local target="$1"
  case "$target" in
    llama) tmux attach -t "$TMUX_SESSION" \; select-window -t llama ;;
    flask) tmux attach -t "$TMUX_SESSION" \; select-window -t flask ;;
    *) tmux attach -t "$TMUX_SESSION" || true ;;
  esac
}

# ---- CLI parsing ----
case "${1:-help}" in
  start-llama|start_llama|start-llama)
    # prefer tmux if installed
    if command -v tmux >/dev/null 2>&1; then start_llama_tmux; else start_llama_nohup; fi
    ;;

  start-flask|start_flask)
    if command -v tmux >/dev/null 2>&1; then start_flask_tmux; else start_flask_nohup; fi
    ;;

  start-all|start)
    echo "Starting LLaMA then Flask..."
    if command -v tmux >/dev/null 2>&1; then
      # make sure session exists
      if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        # create session with llama as first window
        tmux new-session -d -s "$TMUX_SESSION" -n llama "bash -lc 'exec \"$HH_LLAMA_BIN\" -m \"$HH_LLAMA_MODEL\" --port \"$HH_LLAMA_PORT\" --threads 4 -c 2048 2>&1 | tee \"$HH_LLAMA_LOG\"'"
        sleep 0.6
        # add flask window
        tmux new-window -t "$TMUX_SESSION" -n flask "bash -lc 'source \"$HH_VENV_ACTIVATE\" 2>/dev/null || true; exec python \"$HH_FLASK_APP\" 2>&1 | tee \"$HH_FLASK_LOG\"'"
      else
        start_llama_tmux
        start_flask_tmux
      fi
    else
      start_llama_nohup
      start_flask_nohup
    fi
    ;;

  stop-llama|stop_llama)
    stop_llama
    ;;

  stop-flask|stop_flask)
    stop_flask
    ;;

  stop-all|stop)
    echo "Stopping Flask then LLaMA..."
    stop_flask
    stop_llama
    ;;

  status|--status|-s)
    _status
    ;;

  attach)
    if [ -z "${2:-}" ]; then
      echo "Usage: $0 attach {llama|flask}"
      exit 2
    fi
    _attach "$2"
    ;;

  refresh|restart)
    echo "Refreshing servers: stopping then starting (only configured ports)."
    stop_flask
    stop_llama
    sleep 0.6
    # restart in same mode (prefer tmux if available)
    if command -v tmux >/dev/null 2>&1; then
      # remove any dead tmux session to avoid conflicts
      tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
      tmux new-session -d -s "$TMUX_SESSION" -n llama "bash -lc 'exec \"$HH_LLAMA_BIN\" -m \"$HH_LLAMA_MODEL\" --port \"$HH_LLAMA_PORT\" --threads 4 -c 2048 2>&1 | tee \"$HH_LLAMA_LOG\"'"
      sleep 0.4
      tmux new-window -t "$TMUX_SESSION" -n flask "bash -lc 'source \"$HH_VENV_ACTIVATE\" 2>/dev/null || true; exec python \"$HH_FLASK_APP\" 2>&1 | tee \"$HH_FLASK_LOG\"'"
    else
      start_llama_nohup
      start_flask_nohup
    fi
    echo "Done."
    ;;

  help|--help|-h|*)
    cat <<EOF
Usage: $0 <command>

Commands:
  start-llama       Start only LLaMA server
  start-flask       Start only Flask app
  start-all         Start LLaMA then Flask (prefers tmux if installed)
  stop-llama        Stop only LLaMA
  stop-flask        Stop only Flask
  stop-all          Stop Flask then LLaMA
  status            Show status + pids + log file locations
  attach <llama|flask>  Attach to tmux session window (if tmux present)
  refresh|restart   Stop then start both (clean restart)
  help              Show this message

Notes:
 - The script prefers tmux if available. If tmux is not installed it falls back to nohup.
 - Logs are written to: $HH_LLAMA_LOG and $HH_FLASK_LOG
 - Adjust HH_* environment variables in your ~/.bashrc if needed.

EOF
    ;;
esac

exit 0
