#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
SNAPSHOT_DIR="$LOG_DIR/data_snapshots"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -f "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
fi

mkdir -p "$SNAPSHOT_DIR"
cd "$PROJECT_ROOT"

TS="$(date -u +"%Y%m%dT%H%M%SZ")"

INVEST_FORCE=1 PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_ROOT/scripts/run_scheduled_trader.sh"

if [[ -f "$PROJECT_ROOT/artifacts/mag7_analysis.json" ]]; then
  cp "$PROJECT_ROOT/artifacts/mag7_analysis.json" "$SNAPSHOT_DIR/mag7_analysis_$TS.json"
elif [[ -f "$PROJECT_ROOT/data/mag7_analysis.json" ]]; then
  cp "$PROJECT_ROOT/data/mag7_analysis.json" "$SNAPSHOT_DIR/mag7_analysis_$TS.json"
fi

if [[ -f "$PROJECT_ROOT/artifacts/mag7_reallocation_plan.json" ]]; then
  cp "$PROJECT_ROOT/artifacts/mag7_reallocation_plan.json" "$SNAPSHOT_DIR/mag7_reallocation_plan_$TS.json"
fi

if [[ -f "$PROJECT_ROOT/data/investment_action.json" ]]; then
  cp "$PROJECT_ROOT/data/investment_action.json" "$SNAPSHOT_DIR/investment_action_$TS.json"
fi

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Cron run completed (snapshot $TS)" >> "$LOG_DIR/cron.log"
