#!/usr/bin/env bash
# Wraps scheduled_trader.py. Use SCHEDULED_TRADER_MODE=mag7|invest|close.
set -e
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  [ -x .venv/bin/python ] && PYTHON_BIN=.venv/bin/python
  [ -x venv/bin/python ] && PYTHON_BIN=venv/bin/python
  [ -z "$PYTHON_BIN" ] && PYTHON_BIN=python3
fi
mkdir -p logs artifacts data
export INVEST_FORCE=1

if [ "${SCHEDULED_TRADER_MODE}" = "close" ]; then
  $PYTHON_BIN scheduled_trader.py --close 2>&1 | tee -a logs/scheduled_trader.log
elif [ "${SCHEDULED_TRADER_MODE}" = "mag7" ] || [ "${REALLOCATE_FULL}" = "1" ]; then
  $PYTHON_BIN scheduled_trader.py --mag7 2>&1 | tee -a logs/scheduled_trader.log
else
  $PYTHON_BIN scheduled_trader.py --invest 2>&1 | tee -a logs/scheduled_trader.log
fi
