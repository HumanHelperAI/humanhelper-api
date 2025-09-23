cat > run_dev.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PORT=${1:-5000}
APP_CMD=${APP_CMD:-"python3 app.py"}
NGROK_BIN=${NGROK_BIN:-"$HOME/bin/ngrok"}
LOGDIR=${LOGDIR:-"$HOME/hh_logs"}
mkdir -p "$LOGDIR"

# stop known processes
pkill -f "python3 app.py" || true
sleep 0.3

# start app
echo "Starting app: $APP_CMD (port $PORT)"
nohup $APP_CMD > "$LOGDIR/app.log" 2>&1 &
sleep 0.6

# check app health
if curl -sS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "App is responding on port $PORT"
else
  echo "App not responding on port $PORT; check $LOGDIR/app.log"
  tail -n 200 "$LOGDIR/app.log"
  exit 1
fi

# start ngrok
echo "Starting ngrok -> http://127.0.0.1:$PORT"
$NGROK_BIN http "$PORT" > "$LOGDIR/ngrok.log" 2>&1 &
sleep 1

# print public URL
sleep 1
echo "ngrok tunnels:"
curl -s http://127.0.0.1:4040/api/tunnels || true
echo "tailing logs: (press Ctrl+C to stop)"
tail -F "$LOGDIR/app.log" "$LOGDIR/ngrok.log"
EOF

chmod +x run_dev.sh
# run it:
./run_dev.sh 5000
