#!/usr/bin/env bash
# Safe update for FPL Robo on the Pi.
# Pulls origin/2026-2027 and restarts daemons only when idle / outside
# the near-deadline window (unless --force).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="2026-2027"
REMOTE="origin"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/update.log"
LOCK_FILE="$ROOT/data/run.lock"
FORCE=0
NEAR_DEADLINE_HOURS=2

mkdir -p "$LOG_DIR"

log() {
  local msg="[$(date -Is)] $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

usage() {
  echo "Usage: $0 [--force]"
  echo "  Safe-pull $BRANCH and restart fpl-robo + fpl-robo-auth."
  echo "  Refuses near deadline / while run.lock is held unless --force."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

lock_held_by_live_process() {
  [[ -f "$LOCK_FILE" ]] || return 1
  local pid
  pid="$("$PYTHON" - <<'PY' "$LOCK_FILE"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
    print(int(data.get("pid") or 0))
except Exception:
    print(0)
PY
)"
  pid_alive "$pid"
}

hours_to_deadline() {
  "$PYTHON" - <<'PY'
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.runtime.deadline import hours_to_deadline, next_deadline
try:
    client = FPLClient()
    # Prefer authenticated session if present; public bootstrap is enough for deadline.
    client.load_session("data/session.json")
    deadline = next_deadline(client.bootstrap_static())
    if deadline is None:
        print("")
    else:
        print(f"{hours_to_deadline(deadline['deadline_utc']):.6f}")
except Exception:
    print("")
PY
}

log "Update check starting (force=$FORCE)"

git fetch "$REMOTE" "$BRANCH" >>"$LOG_FILE" 2>&1

LOCAL="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse "$REMOTE/$BRANCH")"
if [[ "$LOCAL" == "$REMOTE_SHA" ]]; then
  log "Already up to date at $LOCAL"
  exit 0
fi

log "Update available: $LOCAL -> $REMOTE_SHA"

if lock_held_by_live_process; then
  if [[ "$FORCE" -eq 1 ]]; then
    log "WARNING: run.lock held by live process; continuing because --force"
  else
    log "Skip: run.lock held by live process (daemon busy)"
    exit 0
  fi
fi

HOURS="$(hours_to_deadline || true)"
if [[ -n "$HOURS" ]]; then
  # bash can't compare floats natively; use python.
  NEAR="$("$PYTHON" - <<PY
h = float("$HOURS")
print("1" if 0 < h <= float("$NEAR_DEADLINE_HOURS") else "0")
PY
)"
  if [[ "$NEAR" == "1" ]]; then
    if [[ "$FORCE" -eq 1 ]]; then
      log "WARNING: within ${NEAR_DEADLINE_HOURS}h of deadline (${HOURS}h); continuing because --force"
    else
      log "Skip: within ${NEAR_DEADLINE_HOURS}h of deadline (${HOURS}h left)"
      exit 0
    fi
  fi
else
  log "Could not determine hours to deadline; proceeding cautiously"
fi

# Stay on tracking branch and fast-forward only.
git checkout "$BRANCH" >>"$LOG_FILE" 2>&1
git pull --ff-only "$REMOTE" "$BRANCH" >>"$LOG_FILE" 2>&1
NEW_SHA="$(git rev-parse HEAD)"
log "Pulled $NEW_SHA"

# Reinstall/reload units from repo so TimeoutStopSec etc stay current.
if [[ -f /etc/systemd/system/fpl-robo.service ]] || systemctl list-unit-files | grep -q '^fpl-robo'; then
  sudo cp "$ROOT/deploy/fpl-robo.service" /etc/systemd/system/fpl-robo.service
  sudo cp "$ROOT/deploy/fpl-robo-auth.service" /etc/systemd/system/fpl-robo-auth.service
  if [[ -f "$ROOT/deploy/fpl-robo-update.service" ]]; then
    sudo cp "$ROOT/deploy/fpl-robo-update.service" /etc/systemd/system/fpl-robo-update.service
    sudo cp "$ROOT/deploy/fpl-robo-update.timer" /etc/systemd/system/fpl-robo-update.timer
  fi
  sudo systemctl daemon-reload
fi

sudo systemctl restart fpl-robo-auth.service fpl-robo.service
log "Restarted fpl-robo-auth.service and fpl-robo.service"
systemctl is-active fpl-robo-auth.service fpl-robo.service | tee -a "$LOG_FILE"
log "Update complete"
