#!/usr/bin/env bash
# Supervised Cloudflare Quick Tunnel for the FPL Robo dashboard.
# cloudflared is PID 1 of this script's process group (via exec) so systemd
# Restart=always recovers cleanly. A side watcher writes the public URL file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL_FILE="${CLOUDFLARE_TUNNEL_URL_FILE:-$ROOT/data/cloudflare_tunnel_url.txt}"
LOG_FILE="${CLOUDFLARED_LOG_FILE:-$ROOT/logs/cloudflared_quick.log}"
TARGET="${CLOUDFLARED_TARGET:-http://127.0.0.1:8787}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-/usr/local/bin/cloudflared}"

mkdir -p "$(dirname "$URL_FILE")" "$(dirname "$LOG_FILE")"
: >"$URL_FILE"

# Watch the log for the trycloudflare URL and mirror it to URL_FILE.
(
  # Give cloudflared a moment to open the log via tee below... we append ourselves.
  touch "$LOG_FILE"
  tail -n0 -F "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    if [[ "$line" =~ (https://[a-zA-Z0-9-]+\.trycloudflare\.com) ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}" >"$URL_FILE"
    fi
  done
) &
watcher_pid=$!

cleanup() {
  kill "$watcher_pid" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# Line-buffer into the log; cloudflared stays the foreground job we wait on.
stdbuf -oL -eL "$CLOUDFLARED_BIN" tunnel --url "$TARGET" >>"$LOG_FILE" 2>&1 &
cf_pid=$!
trap 'kill "$cf_pid" "$watcher_pid" 2>/dev/null || true; wait "$cf_pid" 2>/dev/null || true' EXIT TERM INT
wait "$cf_pid"
exit $?
