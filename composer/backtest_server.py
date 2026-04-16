#!/usr/bin/env python3
"""
Backtest server — FastAPI.
Run with: python3 backtest_server.py
Listens on http://localhost:8001
"""
import math
import os
from datetime import date, timedelta
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Backtest Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / response models ─────────────────────────────────────────────────

class Condition(BaseModel):
    id: str
    indicator: str
    period: int = 14
    operator: str
    compareType: str  # "number" | "indicator"
    value: float = 0.0
    compareIndicator: Optional[str] = None
    comparePeriod: Optional[int] = None

class ConditionGroup(BaseModel):
    logic: str  # "AND" | "OR"
    conditions: List[Condition]

class Action(BaseModel):
    id: str
    type: str  # "buy" | "sell" | "hold_cash"
    asset: str = ""
    pct: float = 100.0

class Rule(BaseModel):
    id: str
    conditions: ConditionGroup
    thenActions: List[Action]
    elseActions: List[Action]

class Strategy(BaseModel):
    name: str
    description: str = ""
    rules: List[Rule]

class BacktestRequest(BaseModel):
    strategy: Strategy
    start: str = Field(
        default_factory=lambda: (date.today() - timedelta(days=365)).isoformat()
    )
    end: str = Field(default_factory=lambda: date.today().isoformat())
    capital: float = 100_000.0

# ── Market data ───────────────────────────────────────────────────────────────

def fetch_bars(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    key = os.getenv("ALPACA_KEY", "")
    secret = os.getenv("ALPACA_SECRET", "")
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        params: dict = {
            "timeframe": "1Day",
            "start": start,
            "end": end,
            "adjustment": "split",
            "limit": 1000,
        }
        bars: list = []
        while True:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            if not r.ok:
                break
            data = r.json()
            bars.extend(data.get("bars", []))
            token = data.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        if not bars:
            continue
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["t"]).dt.date
        df = df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        result[symbol] = df
    return result

# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))

def _sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()

def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()

def _macd_hist(close: pd.Series) -> pd.Series:
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    return line - signal

def _bbands(close: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + 2 * std, mid - 2 * std

def indicator_at(df: pd.DataFrame, indicator: str, period: int, i: int) -> Optional[float]:
    if i < 0 or i >= len(df):
        return None
    close = df["close"]
    if indicator == "PRICE":
        return float(close.iloc[i])
    elif indicator == "RSI":
        s = _rsi(close, period)
    elif indicator == "SMA":
        s = _sma(close, period)
    elif indicator == "EMA":
        s = _ema(close, period)
    elif indicator == "MACD":
        s = _macd_hist(close)
    elif indicator == "BBANDS_UPPER":
        s, _ = _bbands(close, period)
    elif indicator == "BBANDS_LOWER":
        _, s = _bbands(close, period)
    else:
        return None
    v = s.iloc[i]
    return float(v) if not math.isnan(v) else None

# ── Condition evaluation ──────────────────────────────────────────────────────

def eval_condition(
    bars: dict[str, pd.DataFrame],
    cond: Condition,
    idx_map: dict[str, int],
    symbol: str,
) -> bool:
    df = bars.get(symbol)
    if df is None:
        return False
    i = idx_map.get(symbol, -1)
    if i < 0:
        return False

    left = indicator_at(df, cond.indicator, cond.period, i)
    if left is None:
        return False

    if cond.compareType == "number":
        right = cond.value
    else:
        ci = cond.compareIndicator or "SMA"
        cp = cond.comparePeriod or 200
        right = indicator_at(df, ci, cp, i)
        if right is None:
            return False

    op = cond.operator
    if op == ">":
        return left > right
    elif op == "<":
        return left < right
    elif op == ">=":
        return left >= right
    elif op == "<=":
        return left <= right
    elif op in ("crosses_above", "crosses_below"):
        if i < 1:
            return False
        lp = indicator_at(df, cond.indicator, cond.period, i - 1)
        rp = (
            cond.value
            if cond.compareType == "number"
            else indicator_at(df, cond.compareIndicator or "SMA", cond.comparePeriod or 200, i - 1)
        )
        if lp is None or rp is None:
            return False
        if op == "crosses_above":
            return lp <= rp and left > right
        return lp >= rp and left < right
    return False

def eval_group(
    bars: dict[str, pd.DataFrame],
    group: ConditionGroup,
    idx_map: dict[str, int],
    symbol: str,
) -> bool:
    results = [eval_condition(bars, c, idx_map, symbol) for c in group.conditions]
    return all(results) if group.logic == "AND" else any(results)

# ── Portfolio simulation ──────────────────────────────────────────────────────

def run_backtest(
    strategy: Strategy,
    bars: dict[str, pd.DataFrame],
    capital: float,
):
    all_dates = sorted(set().union(*(set(df.index) for df in bars.values())))
    cash = capital
    positions: dict[str, float] = {}  # symbol → shares
    equity_curve: list[dict] = []
    trades: list[dict] = []

    # Primary symbol = first buy-target in the strategy
    primary = next(
        (
            a.asset.upper()
            for rule in strategy.rules
            for a in rule.thenActions
            if a.type in ("buy", "sell") and a.asset
        ),
        list(bars.keys())[0] if bars else None,
    )

    for dt in all_dates:
        # Build index map for each symbol on this date
        idx_map: dict[str, int] = {}
        for sym, df in bars.items():
            dates_list = list(df.index)
            if dt in df.index:
                idx_map[sym] = dates_list.index(dt)

        if primary and primary in bars:
            for rule in strategy.rules:
                matched = eval_group(bars, rule.conditions, idx_map, primary)
                actions = rule.thenActions if matched else rule.elseActions

                for action in actions:
                    sym = action.asset.upper() if action.asset else ""
                    sym_df = bars.get(sym)

                    if action.type == "hold_cash":
                        # Liquidate all positions
                        for s in list(positions.keys()):
                            sdf = bars.get(s)
                            if sdf is not None and dt in sdf.index:
                                p = float(sdf.loc[dt, "close"])
                                cash += positions[s] * p
                                trades.append(
                                    {"date": str(dt), "symbol": s, "side": "sell",
                                     "price": round(p, 2), "shares": round(positions[s], 4)}
                                )
                        positions.clear()
                        break

                    if not sym or sym_df is None or dt not in sym_df.index:
                        continue
                    price = float(sym_df.loc[dt, "close"])

                    # Current total equity for sizing
                    total_eq = cash + sum(
                        positions.get(s2, 0) * float(bars[s2].loc[dt, "close"])
                        for s2 in positions
                        if s2 in bars and dt in bars[s2].index
                    )
                    target_val = total_eq * (action.pct / 100.0)
                    curr_val = positions.get(sym, 0) * price
                    delta = target_val - curr_val

                    if action.type == "buy" and delta > 1.0 and cash >= delta:
                        shares = delta / price
                        positions[sym] = positions.get(sym, 0) + shares
                        cash -= delta
                        trades.append(
                            {"date": str(dt), "symbol": sym, "side": "buy",
                             "price": round(price, 2), "shares": round(shares, 4)}
                        )
                    elif action.type in ("buy", "sell") and delta < -1.0:
                        shares_sell = min(abs(delta) / price, positions.get(sym, 0))
                        positions[sym] = positions.get(sym, 0) - shares_sell
                        cash += shares_sell * price
                        trades.append(
                            {"date": str(dt), "symbol": sym, "side": "sell",
                             "price": round(price, 2), "shares": round(shares_sell, 4)}
                        )
                break  # Only first rule that matched/didn't match gets executed

        # Daily equity snapshot
        total = cash + sum(
            positions.get(s, 0) * float(bars[s].loc[dt, "close"])
            for s in positions
            if s in bars and dt in bars[s].index
        )
        equity_curve.append({"date": str(dt), "equity": round(total, 2)})

    return equity_curve, trades

def compute_metrics(equity_curve: list, capital: float) -> dict:
    if not equity_curve:
        return {}
    equities = [p["equity"] for p in equity_curve]
    total_return = (equities[-1] - capital) / capital * 100

    rets = pd.Series(equities).pct_change().dropna()
    sharpe = 0.0
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float((rets.mean() / rets.std()) * math.sqrt(252))

    ser = pd.Series(equities)
    drawdown = (ser - ser.cummax()) / ser.cummax()
    max_dd = float(drawdown.min()) * 100

    win_rate = float((rets > 0).mean() * 100) if len(rets) > 0 else 0.0

    return {
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "final_equity": round(equities[-1], 2),
    }

# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    symbols = {
        a.asset.upper()
        for rule in req.strategy.rules
        for a in rule.thenActions + rule.elseActions
        if a.type in ("buy", "sell") and a.asset
    }
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols found in strategy actions")

    # Fetch strategy symbols + SPY for benchmark
    all_symbols = list(symbols | {"SPY"})
    try:
        bars = fetch_bars(all_symbols, req.start, req.end)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market data fetch failed: {e}")

    # Split SPY out before simulation
    spy_df = bars.pop("SPY", None)
    strategy_bars = {k: v for k, v in bars.items() if k in symbols}

    if not strategy_bars:
        raise HTTPException(
            status_code=400,
            detail=f"No data returned for {symbols} between {req.start} and {req.end}",
        )

    equity_curve, trades = run_backtest(req.strategy, strategy_bars, req.capital)
    metrics = compute_metrics(equity_curve, req.capital)

    # Build SPY benchmark normalized to starting capital
    benchmark_curve: list[dict] = []
    if spy_df is not None and not spy_df.empty:
        spy_start = float(spy_df.iloc[0]["close"])
        for dt, row in spy_df.iterrows():
            benchmark_curve.append({
                "date": str(dt),
                "equity": round(req.capital * float(row["close"]) / spy_start, 2),
            })

    return {
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "trades": trades[:500],
        "metrics": metrics,
        "symbols": list(symbols),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
