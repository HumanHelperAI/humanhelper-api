#!/usr/bin/env bash
set -euo pipefail

# port your flask app listens on
PORT=${1:-5000}

# ngrok binary
NGROK_BIN=${NGROK_BIN:-"$HOME/bin/ngrok"}

# resolv override file (private to user)
RESOLV_FILE="$HOME/.ngrok_resolv.conf"
echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1" > "$RESOLV_FILE"

# Ensure go uses Go resolver (avoids broken local ::1 DNS)
export GODEBUG=netdns=go

# Helpful env debug output
printf "Starting ngrok (port=%s) with GODEBUG=%s\n" "$PORT" "$GODEBUG"
printf "Using resolv file: %s\n" "$RESOLV_FILE"

# start ngrok in foreground so you can see the logs
# also pass --log=stdout so we can tail it or capture it
"$NGROK_BIN" http "$PORT" --log=stdout
