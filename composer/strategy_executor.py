#!/usr/bin/env python3
"""
Strategy Executor — runs every weekday at market open (9:31 AM ET).
Reads active strategies from Supabase, evaluates conditions against live
Alpaca data, and places orders when conditions are met.

Usage:
  python3 strategy_executor.py          # runs on a schedule continuously
  python3 strategy_executor.py --now    # evaluate and trade immediately, then exit
"""
import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import schedule
import time
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ── Env checks ────────────────────────────────────────────────────────────────

ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
ALPACA_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# ── Backtest helpers (reuse logic from backtest_server.py) ────────────────────
# We import the indicator and evaluation functions directly.

import sys
sys.path.insert(0, os.path.dirname(__file__))
from backtest_server import (
    fetch_bars,
    eval_group,
    ConditionGroup,
    Condition,
    Action,
    Rule,
    Strategy,
)
from pydantic import TypeAdapter
import json

# ── Alpaca trading ────────────────────────────────────────────────────────────

import requests

def alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

def get_account():
    r = requests.get(f"{ALPACA_BASE}/v2/account", headers=alpaca_headers(), timeout=10)
    r.raise_for_status()
    return r.json()

def get_positions():
    r = requests.get(f"{ALPACA_BASE}/v2/positions", headers=alpaca_headers(), timeout=10)
    r.raise_for_status()
    return {p["symbol"]: p for p in r.json()}

def place_order(symbol: str, qty: float, side: str):
    payload = {
        "symbol": symbol,
        "qty": str(round(qty, 4)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    r = requests.post(
        f"{ALPACA_BASE}/v2/orders",
        headers={**alpaca_headers(), "Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    log.info(f"ORDER placed: {side} {qty:.4f} {symbol} → {r.json().get('id')}")
    return r.json()

# ── Supabase helpers ──────────────────────────────────────────────────────────

def get_active_strategies():
    """Fetch all active strategies from Supabase using service role key."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        log.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set — skipping DB fetch")
        return []
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{SUPABASE_URL}/rest/v1/active_strategies"
    params = {
        "select": "id,strategy_id,user_id,status,strategies(logic_json)",
        "status": "eq.active",
    }
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def update_last_run(active_id: str):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{SUPABASE_URL}/rest/v1/active_strategies"
    params = {"id": f"eq.{active_id}"}
    requests.patch(
        url,
        headers=headers,
        params=params,
        json={"last_run_at": datetime.now().isoformat()},
        timeout=10,
    )

def log_run(strategy_id: str, user_id: str, status: str, signals_fired: int,
            orders_placed: list, error_msg: str | None = None):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    url = f"{SUPABASE_URL}/rest/v1/strategy_runs"
    try:
        requests.post(
            url,
            headers=headers,
            json={
                "strategy_id": strategy_id,
                "user_id": user_id,
                "status": status,
                "signals_fired": signals_fired,
                "orders_placed": orders_placed,
                "error_msg": error_msg,
            },
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Failed to log run: {e}")

# ── Core execution ────────────────────────────────────────────────────────────

def execute_strategy(active: dict):
    """Evaluate one active strategy against live data and place orders."""
    strategy_data = active.get("strategies") or {}
    logic_json = strategy_data.get("logic_json")
    strategy_id = active.get("strategy_id", "")
    user_id = active.get("user_id", "")
    orders_placed: list = []
    signals_fired = 0

    if not logic_json:
        log.warning(f"No logic_json for active_strategy {active['id']}")
        log_run(strategy_id, user_id, "error", 0, [], "No logic_json found")
        return

    # Parse strategy
    try:
        strategy = Strategy(**logic_json)
    except Exception as e:
        log.error(f"Failed to parse strategy: {e}")
        log_run(strategy_id, user_id, "error", 0, [], str(e))
        return

    # Collect symbols
    symbols = set()
    for rule in strategy.rules:
        for action in rule.thenActions + rule.elseActions:
            if action.type in ("buy", "sell") and action.asset:
                symbols.add(action.asset.upper())

    if not symbols:
        return

    # Fetch recent bars (last 250 days for indicator warmup)
    start = (date.today() - timedelta(days=250)).isoformat()
    end = date.today().isoformat()
    try:
        bars = fetch_bars(list(symbols), start, end)
    except Exception as e:
        log.error(f"Failed to fetch bars: {e}")
        return

    if not bars:
        log.warning(f"No bars returned for {symbols}")
        return

    # Primary symbol
    primary = next(
        (a.asset.upper() for rule in strategy.rules for a in rule.thenActions if a.type in ("buy","sell") and a.asset),
        list(bars.keys())[0]
    )
    if primary not in bars:
        return

    # Build index map for latest date
    idx_map = {}
    for sym, df in bars.items():
        idx_map[sym] = len(df) - 1  # latest row

    # Get account state
    try:
        account = get_account()
        positions = get_positions()
        equity = float(account["equity"])
        buying_power = float(account["buying_power"])
    except Exception as e:
        log.error(f"Failed to fetch Alpaca account: {e}")
        return

    log.info(f"Strategy '{strategy.name}' | equity=${equity:,.0f} | symbols={symbols}")

    # Evaluate rules
    for rule in strategy.rules:
        matched = eval_group(bars, rule.conditions, idx_map, primary)
        actions = rule.thenActions if matched else rule.elseActions
        log.info(f"  Rule matched={matched} → executing {len(actions)} action(s)")

        for action in actions:
            sym = action.asset.upper() if action.asset else ""
            if action.type == "hold_cash":
                # Liquidate positions
                for s, pos in list(positions.items()):
                    qty = float(pos["qty"])
                    if qty > 0:
                        log.info(f"  hold_cash: selling {qty} {s}")
                        try:
                            place_order(s, qty, "sell")
                        except Exception as e:
                            log.error(f"  sell failed: {e}")
                break

            if not sym or sym not in bars:
                continue

            sym_df = bars[sym]
            current_price = float(sym_df.iloc[-1]["close"])
            target_value = equity * (action.pct / 100.0)
            current_pos = positions.get(sym)
            current_value = float(current_pos["market_value"]) if current_pos else 0.0
            delta = target_value - current_value

            if action.type == "buy" and delta > 50 and buying_power >= delta:
                shares = delta / current_price
                log.info(f"  buy {shares:.4f} {sym} @ ~${current_price:.2f} (target ${target_value:,.0f})")
                try:
                    result = place_order(sym, shares, "buy")
                    orders_placed.append({"side": "buy", "symbol": sym, "shares": round(shares, 4), "price": round(current_price, 2), "order_id": result.get("id")})
                    signals_fired += 1
                except Exception as e:
                    log.error(f"  buy failed: {e}")

            elif action.type == "sell" and delta < -50 and current_pos:
                shares = abs(delta) / current_price
                shares = min(shares, float(current_pos.get("qty", 0)))
                if shares > 0:
                    log.info(f"  sell {shares:.4f} {sym} @ ~${current_price:.2f}")
                    try:
                        result = place_order(sym, shares, "sell")
                        orders_placed.append({"side": "sell", "symbol": sym, "shares": round(shares, 4), "price": round(current_price, 2), "order_id": result.get("id")})
                        signals_fired += 1
                    except Exception as e:
                        log.error(f"  sell failed: {e}")
        break  # First matching rule wins

    update_last_run(active["id"])
    log_run(strategy_id, user_id, "completed", signals_fired, orders_placed)


def run_all():
    """Fetch and execute all active strategies."""
    log.info("=== Strategy Executor run started ===")
    try:
        actives = get_active_strategies()
    except Exception as e:
        log.error(f"Failed to fetch active strategies from Supabase: {e}")
        return

    if not actives:
        log.info("No active strategies found.")
        return

    log.info(f"Found {len(actives)} active strategy/strategies")
    for active in actives:
        try:
            execute_strategy(active)
        except Exception as e:
            log.error(f"Error executing strategy {active.get('id')}: {e}")

    log.info("=== Run complete ===")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def is_market_day():
    today = datetime.now(ET)
    return today.weekday() < 5  # Mon–Fri

def scheduled_job():
    if is_market_day():
        run_all()
    else:
        log.info("Market closed today — skipping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Run immediately and exit")
    args = parser.parse_args()

    if args.now:
        run_all()
        sys.exit(0)

    # Schedule at 9:31 AM ET every weekday
    schedule.every().day.at("09:31").do(scheduled_job)
    log.info("Strategy executor scheduled — will run at 09:31 ET every weekday")
    log.info("Run with --now to execute immediately")

    while True:
        schedule.run_pending()
        time.sleep(30)
