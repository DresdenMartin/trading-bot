"""EOD price data and news/premarket placeholders. Uses yfinance for OHLCV."""
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


def fetch_eod(symbols: List[str], output_dir: str = "./data") -> Dict[str, pd.DataFrame]:
    """Fetch end-of-day OHLCV for symbols. Returns dict symbol -> DataFrame with Close, High, Low, Volume."""
    if not symbols:
        return {}
    symbols = [s.strip().upper() for s in symbols if s and str(s).strip()]
    if not symbols:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    try:
        # ~6 months of daily data for indicators (SMA 200, etc.)
        tickers = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                t = tickers.tickers.get(sym) if hasattr(tickers, "tickers") else yf.Ticker(sym)
                if t is None:
                    t = yf.Ticker(sym)
                df = t.history(period="6mo", interval="1d", auto_adjust=True)
                if df is None or df.empty or len(df) < 2:
                    out[sym] = pd.DataFrame()
                    continue
                df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
                if "Close" not in df.columns:
                    out[sym] = pd.DataFrame()
                    continue
                out[sym] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            except Exception:
                out[sym] = pd.DataFrame()
    except Exception:
        for sym in symbols:
            try:
                df = yf.download(sym, period="6mo", interval="1d", group_by=False, auto_adjust=True, progress=False, threads=False)
                if df is None or df.empty:
                    out[sym] = pd.DataFrame()
                else:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    need = ["Open", "High", "Low", "Close", "Volume"]
                    if not all(c in df.columns for c in need):
                        out[sym] = pd.DataFrame()
                    else:
                        out[sym] = df[need].copy()
            except Exception:
                out[sym] = pd.DataFrame()
    return out


def aggregate_news_for_symbol(
    symbol: str, limit: int = 20, hours: float = 24
) -> Tuple[List[Dict[str, Any]], float]:
    """Return (list of article dicts, aggregate sentiment). Placeholder: no news API."""
    return [], 0.0


def fetch_premarket_info(symbol: str) -> Dict[str, Any]:
    """Return premarket info: sector, dollar_volume, spread. Placeholder for real API."""
    try:
        t = yf.Ticker(symbol.strip().upper())
        info = t.info or {}
        sector = info.get("sector") or info.get("industry") or "UNKNOWN"
        cap = info.get("marketCap")
        dollar_volume = float(cap) * 0.01 if cap else 5_000_000.0  # assume enough for guards
        return {"sector": sector, "dollar_volume": dollar_volume, "spread": 0.002}
    except Exception:
        return {"sector": "UNKNOWN", "dollar_volume": 5_000_000.0, "spread": 0.002}


def fetch_alpaca_intraday_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Intraday bars from Alpaca. Placeholder: returns None (no intraday)."""
    return None
