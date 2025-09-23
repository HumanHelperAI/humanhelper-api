#!/bin/bash
set -eu

cd ~/humanhelper/backend
. .venv/bin/activate || true
mkdir -p ~/logs

# start Flask if not running
if ! pgrep -fa "python3 app.py" >/dev/null; then
  nohup python3 app.py > ~/logs/flask.out 2>&1 &
  sleep 0.6
fi

# stop any cloudflared
pkill -f cloudflared || true
sleep 0.2

# start cloudflared quick tunnel in background
GODEBUG=netdns=go \
RES_OPTIONS="ndots:0" \
RESOLV_OVERRIDE="$HOME/.cloudflared_resolv.conf" \
SSL_CERT_FILE="${PREFIX:-/data/data/com.termux/files/usr}/etc/tls/cert.pem" \
"$HOME/bin/cloudflared" tunnel --no-autoupdate --url http://127.0.0.1:5000 \
  > ~/logs/cloudflared.try.log 2>&1 &

sleep 1
echo "Flask logs: ~/logs/flask.out"
echo "Cloudflared logs: ~/logs/cloudflared.try.log"
tail -n 80 ~/logs/cloudflared.try.log || true
