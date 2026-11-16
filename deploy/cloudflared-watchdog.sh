#!/usr/bin/env bash
# Watchdog for the Cloudflare Quick Tunnel. cloudflared can sit "running" forever
# after Cloudflare drops an account-less tunnel, retrying a dead tunnel without
# exiting, so systemd's Restart=always never fires. This probes the public URL
# and restarts the tunnel unit after consecutive failures.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT="${CLOUDFLARED_UNIT:-cloudflared-fpl-robo-quick.service}"
URL_FILE="${CLOUDFLARE_TUNNEL_URL_FILE:-$ROOT/data/cloudflare_tunnel_url.txt}"
LOCAL_URL="${CLOUDFLARED_TARGET:-http://127.0.0.1:8787}"
LOG_FILE="${CLOUDFLARED_WATCHDOG_LOG:-$ROOT/logs/cloudflared_watchdog.log}"
STATE_FILE="${CLOUDFLARED_WATCHDOG_STATE:-/run/fpl-robo-cloudflared-watchdog.failures}"
MAX_FAILURES="${CLOUDFLARED_WATCHDOG_MAX_FAILURES:-2}"
GRACE_SECONDS="${CLOUDFLARED_WATCHDOG_GRACE_SECONDS:-180}"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG_FILE"; }

reset_failures() { rm -f "$STATE_FILE"; }

restart_tunnel() {
  log "restarting $UNIT: $1"
  reset_failures
  systemctl restart "$UNIT"
}

# Not running at all: systemd (or an operator) owns that state, not us.
if ! systemctl is-active --quiet "$UNIT"; then
  reset_failures
  exit 0
fi

# Give a freshly (re)started tunnel time to register and publish its URL.
started_us="$(systemctl show -p ActiveEnterTimestampMonotonic --value "$UNIT")"
now_us="$(awk '{printf "%d", $1 * 1000000}' /proc/uptime)"
if (( (now_us - started_us) / 1000000 < GRACE_SECONDS )); then
  exit 0
fi

# Dashboard itself down: a tunnel restart cannot help.
if ! curl -s -o /dev/null -m 10 "$LOCAL_URL/"; then
  log "local dashboard $LOCAL_URL unreachable; skipping tunnel check"
  exit 0
fi

url="$(head -n1 "$URL_FILE" 2>/dev/null | tr -d '[:space:]')"
if [[ -z "$url" ]]; then
  restart_tunnel "no public URL published after ${GRACE_SECONDS}s"
  exit 0
fi

# Cloudflare answers 530 (error 1033) for a dead tunnel; 000 is no response.
code="$(curl -s -o /dev/null -m 20 -w '%{http_code}' "$url/")"
if [[ "$code" != "000" && "$code" -lt 500 ]]; then
  reset_failures
  exit 0
fi

failures=$(( $(cat "$STATE_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$failures" >"$STATE_FILE"
log "probe $url failed (http $code), consecutive failures: $failures/$MAX_FAILURES"
if (( failures >= MAX_FAILURES )); then
  restart_tunnel "public URL unhealthy for $failures checks"
fi
