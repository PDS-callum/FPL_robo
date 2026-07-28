#!/usr/bin/env bash
# Long-lived Cloudflare Quick Tunnel in tmux (stable URL until this process dies).
# Usage:
#   deploy/start-cloudflare-tmux.sh          # start if not running
#   tmux attach -t fpl-tunnel                # view logs
#   cat data/cloudflare_tunnel_url.txt       # current public URL
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${CLOUDFLARE_TMUX_SESSION:-fpl-tunnel}"
URL_FILE="${CLOUDFLARE_TUNNEL_URL_FILE:-$ROOT/data/cloudflare_tunnel_url.txt}"
LOG_FILE="${CLOUDFLARED_LOG_FILE:-$ROOT/logs/cloudflared_quick.log}"
TARGET="${CLOUDFLARED_TARGET:-http://127.0.0.1:8787}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-/usr/local/bin/cloudflared}"

mkdir -p "$(dirname "$URL_FILE")" "$(dirname "$LOG_FILE")"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed" >&2
  exit 1
fi

if pgrep -x cloudflared >/dev/null 2>&1; then
  echo "cloudflared already running (pid $(pgrep -x cloudflared | head -1))"
  if [[ -s "$URL_FILE" ]]; then
    echo "URL: $(tr -d '\n' <"$URL_FILE")"
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session: $SESSION (attach with: tmux attach -t $SESSION)"
  fi
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

: >"$URL_FILE"

# Run cloudflared in tmux; scrape URL from its output into URL_FILE.
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "stdbuf -oL -eL $CLOUDFLARED_BIN tunnel --url $TARGET 2>&1 | tee -a $LOG_FILE | while IFS= read -r line; do
     echo \"\$line\"
     if [[ \"\$line\" =~ (https://[a-zA-Z0-9-]+\\.trycloudflare\\.com) ]]; then
       printf '%s\\n' \"\${BASH_REMATCH[1]}\" > \"$URL_FILE\"
     fi
   done"

echo "Started tmux session '$SESSION'"
for _ in $(seq 1 30); do
  if [[ -s "$URL_FILE" ]]; then
    echo "URL: $(tr -d '\n' <"$URL_FILE")"
    exit 0
  fi
  sleep 1
done
echo "Tunnel started but URL not captured yet — check: tmux attach -t $SESSION" >&2
exit 0
