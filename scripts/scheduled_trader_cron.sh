#!/usr/bin/env bash
# Cron wrapper: runs the scheduled trader and archives outputs.
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs logs/data_snapshots

# Run the trader (loads .env from project root if present)
bash scripts/run_scheduled_trader.sh || true

# Snapshot outputs with timestamp
TS=$(date +%Y%m%d_%H%M%S)
[ -f artifacts/mag7_analysis.json ] && cp artifacts/mag7_analysis.json "logs/data_snapshots/mag7_analysis_${TS}.json" 2>/dev/null || true
[ -f data/mag7_analysis.json ] && cp data/mag7_analysis.json "logs/data_snapshots/mag7_analysis_${TS}.json" 2>/dev/null || true
[ -f artifacts/investment_action.json ] && cp artifacts/investment_action.json "logs/data_snapshots/investment_action_${TS}.json" 2>/dev/null || true
[ -f data/investment_action.json ] && cp data/investment_action.json "logs/data_snapshots/investment_action_${TS}.json" 2>/dev/null || true

# Log status
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) cron run completed" >> logs/cron.log
