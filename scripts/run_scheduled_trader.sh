#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -f "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
fi

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

export INVEST_FORCE=1

MODE="${SCHEDULED_TRADER_MODE:-invest}"
EXTRA_ARGS=()
if [[ -n "${SCHEDULED_TRADER_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${SCHEDULED_TRADER_ARGS}"
fi

TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
LOG_FILE="$LOG_DIR/scheduled_trader.log"

{
  echo "[$TS] Starting scheduled_trader via automation"
  if [[ "${SCHEDULED_TRADER_ENFORCE_TIME:-0}" == "1" ]]; then
    NOW_ET="$(TZ=America/New_York date +%H:%M)"
    if [[ "$NOW_ET" != "09:00" && "$NOW_ET" != "16:30" ]]; then
      echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Skipping run; ET time $NOW_ET is outside 09:00/16:30 window"
      exit 0
    fi
  fi
  case "$MODE" in
    invest)
      "$PYTHON_BIN" scheduled_trader.py --invest "${EXTRA_ARGS[@]}"
      ;;
    mag7)
      "$PYTHON_BIN" scheduled_trader.py --mag7 "${EXTRA_ARGS[@]}"
      ;;
    close)
      "$PYTHON_BIN" scheduled_trader.py --close "${EXTRA_ARGS[@]}"
      ;;
    *)
      echo "Unknown SCHEDULED_TRADER_MODE: $MODE" >&2
      exit 2
      ;;
  esac
  RC=$?
  END_TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[$END_TS] scheduled_trader completed with exit code $RC"
  exit $RC
} >> "$LOG_FILE" 2>&1
