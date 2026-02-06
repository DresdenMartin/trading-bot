import os
import math
import json
import logging
import time
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta, time, timezone
import zoneinfo
import holidays
import yfinance as yf

from eod_fetcher import fetch_eod, aggregate_news_for_symbol, fetch_premarket_info, fetch_alpaca_intraday_bars
from indicators import add_sma, add_ema, add_rsi, add_macd, add_bbands
import analyze_with_chatgpt
from analyze_with_chatgpt import analyze_summary

import requests

from alerts import send_trade_alert

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')


def env_bool(name, default=False):
    v = os.getenv(name, str(default))
    if v is None:
        v = str(default)
    return v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def debug_enabled():
    return os.getenv('DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'y')


def _tprint(label: str, start_ts: float):
    if debug_enabled():
        dur = datetime.now().timestamp() - start_ts
        print(f"[DEBUG] {label}: {dur:.2f}s")


DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA']


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v and str(v).strip() else default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v and str(v).strip() else default


DEFAULT_TOP_ALLOCATE_COUNT = _env_int('TOP_ALLOCATE_COUNT', 3)
TRADE_COOLDOWN_PATH = os.path.join('data', 'trade_cooldowns.json')
COOLDOWN_MINUTES = _env_float('TRADE_COOLDOWN_MINUTES', 30.0)
EARNINGS_BLACKOUT_HOURS = _env_float('EARNINGS_BLACKOUT_HOURS', 24.0)
RISK_PER_TRADE = _env_float('RISK_PER_TRADE', 0.01)
ATR_STOP_MULT = _env_float('TRADE_ATR_STOP_MULT', 1.0)
ATR_TP_MULT = _env_float('TRADE_ATR_TP_MULT', 2.0)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or df.empty or len(df) < period + 1:
        return None
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period).mean()
    atr_value = atr_series.iloc[-1]
    if pd.isna(atr_value):
        return None
    return float(atr_value)


def _compute_indicator_snapshot(df: pd.DataFrame, intraday_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    indicators: Dict[str, Optional[float]] = {}
    if df is None or df.empty:
        return indicators
    close = float(df['Close'].iloc[-1])
    indicators['close'] = close
    for window in (20, 50, 200):
        series = df['Close'].rolling(window).mean()
        indicators[f'sma{window}'] = float(series.iloc[-1]) if len(series.dropna()) else None
    ema21 = df['Close'].ewm(span=21, adjust=False).mean()
    indicators['ema21'] = float(ema21.iloc[-1]) if len(ema21.dropna()) else None
    macd_df = add_macd(df)
    indicators['macd'] = float(macd_df['MACD'].iloc[-1]) if len(macd_df['MACD'].dropna()) else None
    indicators['macd_signal'] = float(macd_df['MACD_Signal'].iloc[-1]) if len(macd_df['MACD_Signal'].dropna()) else None
    indicators['macd_hist'] = float(macd_df['MACD_Hist'].iloc[-1]) if len(macd_df['MACD_Hist'].dropna()) else None
    rsi_series = add_rsi(df, 14)
    indicators['rsi14'] = float(rsi_series.iloc[-1]) if len(rsi_series.dropna()) else None
    atr_value = _compute_atr(df)
    indicators['atr'] = atr_value
    indicators['atr_percent'] = float((atr_value / close) * 100) if atr_value and close else None
    volume_series = df['Volume'].dropna() if 'Volume' in df.columns else pd.Series(dtype=float)
    indicators['recent_volume'] = float(volume_series.iloc[-1]) if len(volume_series) else None
    if len(volume_series) >= 30:
        ref = volume_series.iloc[-31:-1].median() if len(volume_series) > 30 else volume_series.median()
        if ref and ref > 0:
            indicators['rvol'] = float(volume_series.iloc[-1] / ref)
        else:
            indicators['rvol'] = None
    else:
        indicators['rvol'] = None
    if intraday_df is not None and not intraday_df.empty:
        vol_sum = intraday_df['volume'].sum()
        if vol_sum > 0:
            indicators['session_vwap'] = float((intraday_df['close'] * intraday_df['volume']).sum() / vol_sum)
            indicators['session_volume'] = float(vol_sum)
        else:
            indicators['session_vwap'] = None
            indicators['session_volume'] = None
    else:
        indicators['session_vwap'] = None
        indicators['session_volume'] = None
    indicators['above_sma20'] = indicators.get('sma20') is not None and close > indicators['sma20'] if indicators.get('sma20') is not None else None
    indicators['above_sma50'] = indicators.get('sma50') is not None and close > indicators['sma50'] if indicators.get('sma50') is not None else None
    indicators['above_sma200'] = indicators.get('sma200') is not None and close > indicators['sma200'] if indicators.get('sma200') is not None else None
    return indicators


def _load_trade_cooldowns() -> Dict[str, str]:
    try:
        with open(TRADE_COOLDOWN_PATH, 'r') as fh:
            return json.load(fh)
    except Exception:
        return {}


def _check_cooldown(symbol: str) -> Dict[str, Optional[float]]:
    if COOLDOWN_MINUTES <= 0:
        return {'cooldown_active': False, 'minutes_remaining': None}
    data = _load_trade_cooldowns()
    ts = data.get(symbol)
    if not ts:
        return {'cooldown_active': False, 'minutes_remaining': None}
    try:
        last = datetime.fromisoformat(ts)
    except Exception:
        return {'cooldown_active': False, 'minutes_remaining': None}
    now = datetime.now(timezone.utc)
    minutes_elapsed = (now - last).total_seconds() / 60.0
    if minutes_elapsed < COOLDOWN_MINUTES:
        return {'cooldown_active': True, 'minutes_remaining': round(COOLDOWN_MINUTES - minutes_elapsed, 2)}
    return {'cooldown_active': False, 'minutes_remaining': None}


def _earnings_within_blackout(symbol: str, window_hours: float = EARNINGS_BLACKOUT_HOURS) -> Dict[str, Optional[str]]:
    if window_hours <= 0:
        return {'earnings_blackout': False, 'earnings_time': None}
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.get_earnings_dates(limit=1)
        if cal is None or cal.empty:
            return {'earnings_blackout': False, 'earnings_time': None}
        next_ts = cal.index[0].to_pydatetime()
        if next_ts.tzinfo is None:
            next_ts = next_ts.replace(tzinfo=timezone.utc)
        now = datetime.now(next_ts.tzinfo)
        delta_hours = (next_ts - now).total_seconds() / 3600.0
        if 0 <= delta_hours <= window_hours:
            return {'earnings_blackout': True, 'earnings_time': next_ts.isoformat()}
    except Exception:
        return {'earnings_blackout': False, 'earnings_time': None}
    return {'earnings_blackout': False, 'earnings_time': None}


def _collect_news_catalysts(articles: List[Dict]) -> List[str]:
    counter = Counter()
    for art in articles or []:
        for tag in art.get('catalyst_tags', []) or []:
            counter[tag] += 1
    return [tag for tag, _ in counter.most_common(5)]


def _score_components(metrics: Dict) -> Dict[str, float]:
    indicators = metrics.get('indicators', {}) or {}
    close = indicators.get('close') or metrics.get('close') or 0.0
    news_sent = float(metrics.get('news_aggregate_sentiment') or 0.0)
    news_conf = float(metrics.get('news_confidence') or 0.0)
    news_articles = metrics.get('news_articles') or []
    top_catalysts = _collect_news_catalysts(news_articles)
    metrics['top_catalysts'] = top_catalysts
    news_count = len(news_articles)

    trend_score = 0.0
    trend_weight = 0.0
    for window, weight in ((20, 0.3), (50, 0.25), (200, 0.2)):
        sma = indicators.get(f'sma{window}')
        if sma:
            trend_weight += weight
            trend_score += weight if close and close > sma else -weight
    trend_component = max(-1.0, min(1.0, trend_score / trend_weight)) if trend_weight else 0.0

    momentum_parts = []
    macd = indicators.get('macd')
    macd_signal = indicators.get('macd_signal')
    if macd is not None and macd_signal is not None:
        denom = abs(macd_signal) + 1e-6
        momentum_parts.append(max(-1.0, min(1.0, (macd - macd_signal) / denom)))
    rsi = indicators.get('rsi14')
    if rsi is not None:
        momentum_parts.append(max(-1.0, min(1.0, (rsi - 50.0) / 20.0)))
    momentum_component = sum(momentum_parts) / len(momentum_parts) if momentum_parts else 0.0

    atr_percent = indicators.get('atr_percent')
    volatility_component = 0.0
    if atr_percent is not None:
        if atr_percent < 1.0:
            volatility_component = -0.4
        elif atr_percent > 12.0:
            volatility_component = -0.3
        elif 2.0 <= atr_percent <= 8.0:
            volatility_component = 0.2

    rvol = indicators.get('rvol')
    rvol_component = 0.0
    if rvol is not None:
        if rvol >= 1.5:
            rvol_component = 0.3
        elif rvol >= 1.1:
            rvol_component = 0.15
        elif rvol <= 0.8:
            rvol_component = -0.2

    news_conf_component = (news_conf - 0.5) * 2.0 if news_conf else 0.0
    news_component = max(-1.0, min(1.0, 0.7 * news_sent + 0.3 * news_conf_component))

    catalyst_component = 0.0
    if top_catalysts:
        if 'earnings/guidance' in top_catalysts:
            catalyst_component += 0.4
        if 'upgrades/downgrades' in top_catalysts:
            catalyst_component += 0.25
        if 'product' in top_catalysts:
            catalyst_component += 0.15
        if 'legal/regulatory' in top_catalysts:
            catalyst_component -= 0.3
        if 'outage' in top_catalysts:
            catalyst_component -= 0.4
        catalyst_component = max(-1.0, min(1.0, catalyst_component))

    return {
        'trend': trend_component,
        'momentum': momentum_component,
        'volatility': volatility_component,
        'rvol': rvol_component,
        'news': news_component,
        'catalysts': catalyst_component,
        'news_count': news_count,
        'news_sent': news_sent,
    }


def _compose_rationale(indicators: Dict[str, Optional[float]], metrics: Dict, components: Dict[str, float], catalysts: List[str]) -> List[str]:
    bullets: List[str] = []
    close = indicators.get('close')
    sma_notes = []
    for window in (20, 50, 200):
        key = f'sma{window}'
        sma = indicators.get(key)
        if sma:
            direction = '>' if close and close > sma else '<'
            sma_notes.append(f"{key.upper()} {direction}")
    if sma_notes:
        bullets.append('Trend alignment ' + ', '.join(sma_notes))
    rsi = indicators.get('rsi14')
    if rsi is not None:
        if rsi > 60:
            bullets.append(f'RSI {rsi:.1f} (bullish)')
        elif rsi < 40:
            bullets.append(f'RSI {rsi:.1f} (bearish)')
        else:
            bullets.append(f'RSI {rsi:.1f} (neutral)')
    macd = indicators.get('macd')
    macd_signal = indicators.get('macd_signal')
    if macd is not None and macd_signal is not None:
        bullets.append('MACD above signal' if macd > macd_signal else 'MACD below signal')
    news_sent = metrics.get('news_aggregate_sentiment', 0.0)
    news_count = components.get('news_count', 0)
    bullets.append(f"News sentiment {news_sent:+.2f} ({int(news_count)} articles)")
    if catalysts:
        bullets.append('Catalysts: ' + ', '.join(catalysts[:3]))
    atr_percent = indicators.get('atr_percent')
    if atr_percent is not None:
        bullets.append(f'ATR% {atr_percent:.2f}')
    rvol = indicators.get('rvol')
    if rvol is not None:
        bullets.append(f'RVOL {rvol:.2f}')
    return bullets[:5]


def _combine_components(components: Dict[str, float]) -> float:
    score_raw = (
        0.35 * components.get('news', 0.0)
        + 0.25 * components.get('trend', 0.0)
        + 0.2 * components.get('momentum', 0.0)
        + 0.1 * components.get('rvol', 0.0)
        + 0.1 * components.get('catalysts', 0.0)
        + 0.05 * components.get('volatility', 0.0)
    )
    return max(-1.0, min(1.0, score_raw))


def _score_items(items: List[Dict]) -> List[Dict]:
    results: List[Dict] = []
    for item in items:
        symbol = item.get('symbol')
        metrics = item.get('metrics', {}) or {}
        components = _score_components(metrics)
        raw_score = _combine_components(components)
        score_pct = int(round((raw_score + 1.0) * 50))

        confidence_pct = int(round((abs(raw_score) * 0.5 + 0.5) * 100))
        decision = 'hold'
        if raw_score >= 0.3:
            decision = 'buy'
        elif raw_score <= -0.3:
            decision = 'sell'

        catalysts = metrics.get('top_catalysts') or []
        indicators = metrics.get('indicators', {}) or {}
        rationale_bullets = _compose_rationale(indicators, metrics, components, catalysts)

        result = {
            'symbol': symbol,
            'score': score_pct,
            'score_raw': raw_score,
            'confidence': confidence_pct,
            'decision': decision,
            'rationale_bullets': rationale_bullets,
            'rationale': '; '.join(rationale_bullets[:3]),
            'details': {
                'news_sent': components.get('news_sent'),
                'news_count': components.get('news_count'),
                'news_confidence': metrics.get('news_confidence'),
                'top_catalysts': catalysts,
                'indicators': indicators,
                'latest_price': metrics.get('latest_price'),
            },
        }
        results.append(result)

    results.sort(key=lambda x: x['score'], reverse=True)
    for idx, entry in enumerate(results, 1):
        entry['rank'] = idx
    return results


def _build_trade_plan(symbol: str, decision: str, indicators: Dict[str, Optional[float]], latest_price: Optional[float], account_equity: Optional[float]) -> Dict[str, object]:
    plan: Dict[str, object] = {'action': decision.upper() if decision else 'HOLD'}
    if latest_price is None or latest_price <= 0:
        plan['notes'] = ['Missing latest price']
        return plan
    atr_value = indicators.get('atr')
    if atr_value is None or atr_value <= 0:
        plan['notes'] = ['ATR unavailable; sizing disabled']
        return plan
    risk_capital = account_equity * max(0.0, RISK_PER_TRADE) if account_equity and account_equity > 0 else None
    position_size = int(max(0, math.floor(risk_capital / atr_value))) if risk_capital else 0
    plan['position_size'] = position_size
    plan['atr'] = atr_value
    plan['atr_percent'] = indicators.get('atr_percent')
    entry = latest_price
    if decision.lower() == 'buy':
        stop = max(0.0, entry - ATR_STOP_MULT * atr_value)
        take_profit = entry + ATR_TP_MULT * atr_value
    elif decision.lower() == 'sell':
        stop = entry + ATR_STOP_MULT * atr_value
        take_profit = entry - ATR_TP_MULT * atr_value
    else:
        stop = None
        take_profit = None
    plan['entry'] = entry
    plan['stop'] = stop
    plan['take_profit'] = take_profit
    plan['time_in_force'] = 'DAY'
    cooldown_info = _check_cooldown(symbol)
    earnings_info = _earnings_within_blackout(symbol)
    plan['blocked'] = {
        'cooldown': cooldown_info,
        'earnings_blackout': earnings_info,
    }
    return plan


def get_symbols():
    s = os.getenv('SYMBOLS', ','.join(DEFAULT_SYMBOLS))
    return [x.strip().upper() for x in s.split(',') if x.strip()]


def discover_top_symbols_via_news(top_n: int = 10, articles_limit: int = 200) -> list:
    """Return the configured Mag Seven universe truncated to the requested length."""
    symbols = get_symbols()
    return symbols[:min(top_n, len(symbols))]


def is_market_holiday(check_date: datetime):
    # uses US federal holidays as approximation (NY market holidays)
    us_holidays = holidays.US()
    return check_date.date() in us_holidays


def in_time_window_for_invest(now_utc: datetime):
    # Market open 9:30 ET; we want 9:15 ET (15 minutes before premarket open) as a window
    et = now_utc.astimezone(zoneinfo.ZoneInfo('America/New_York'))
    # Only weekdays
    if et.weekday() >= 5:
        return False
    if is_market_holiday(et):
        return False
    target = datetime.combine(et.date(), time(9, 15), tzinfo=zoneinfo.ZoneInfo('America/New_York'))
    # allow a 10 minute window
    delta = abs((et - target).total_seconds())
    return delta <= 600


def in_time_window_for_close(now_utc: datetime):
    # Market close 16:00 ET
    et = now_utc.astimezone(zoneinfo.ZoneInfo('America/New_York'))
    if et.weekday() >= 5:
        return False
    if is_market_holiday(et):
        return False
    target = datetime.combine(et.date(), time(16, 0), tzinfo=zoneinfo.ZoneInfo('America/New_York'))
    delta = abs((et - target).total_seconds())
    return delta <= 900


def analyze_and_choose(symbols):
    # fetch price data
    data = fetch_eod(symbols, output_dir='./data')
    candidates = []
    for s, df in data.items():
        if df is None or df.empty:
            continue
        df['SMA_14'] = add_sma(df, 14)
        df['EMA_14'] = add_ema(df, 14)
        df['RSI_14'] = add_rsi(df, 14)
        macd = add_macd(df)
        bb = add_bbands(df)
        df = pd.concat([df, macd, bb], axis=1)
        latest = df.iloc[-1]

        # aggregate news from last 24h
        articles, agg_sent = aggregate_news_for_symbol(s, limit=20)
        # fetch premarket info for indicators & guards
        prem = fetch_premarket_info(s)

        metrics = {
            'close': float(latest['Close']),
            'sma_14': float(latest.get('SMA_14') or 0.0),
            'ema_14': float(latest.get('EMA_14') or 0.0),
            'rsi_14': float(latest.get('RSI_14') or 0.0),
            'news_articles': articles,
            'news_aggregate_sentiment': agg_sent,
            'premarket': prem,
        }
        try:
            analysis = analyze_summary(s, metrics)
        except Exception as e:
            analysis = {'suggested_action': 'hold', 'confidence': 0.0, 'summary': f'analysis failed: {e}'}

    # (no test hooks in production)

        suggested = (analysis.get('suggested_action') or '').lower() if isinstance(analysis, dict) else 'hold'
        conf = float(analysis.get('confidence') or 0.0) if isinstance(analysis, dict) else 0.0
        score = conf + 0.5 * agg_sent
        if suggested == 'buy':
            score += 0.5
        latest_price = float(latest['Close'])
        candidates.append({'symbol': s, 'score': score, 'suggested': suggested, 'confidence': conf, 'news_sentiment': agg_sent, 'analysis': analysis, 'latest_price': latest_price})

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates


def place_buy(symbol, qty):
    key = os.getenv('ALPACA_KEY')
    secret = os.getenv('ALPACA_SECRET')
    if not key or not secret:
        return {'error': 'missing alpaca keys'}
    base = 'https://paper-api.alpaca.markets' if env_bool('ALPACA_PAPER', True) else 'https://api.alpaca.markets'
    headers = {
        'APCA-API-KEY-ID': key,
        'APCA-API-SECRET-KEY': secret,
        'Content-Type': 'application/json',
    }
    extended_hours_enabled = env_bool('EXTENDED_HOURS', False)
    ref_price = None
    if extended_hours_enabled:
        try:
            dfp = fetch_eod([symbol], output_dir='./data').get(symbol)
            if dfp is not None and not dfp.empty:
                ref_price = float(dfp.iloc[-1]['Close'])
        except Exception:
            ref_price = None
    order = {'symbol': symbol, 'qty': qty, 'side': 'buy', 'type': 'market', 'time_in_force': 'day'}
    _maybe_add_extended_hours(order, extended_hours_enabled, reference_price=ref_price)
    try:
        r = requests.post(f"{base}/v2/orders", headers=headers, json=order, timeout=10)
        r.raise_for_status()
        resp = r.json()
        _log_order_result('place_buy', 'buy', order, resp)
        return resp
    except Exception as e:
        resp = {'error': str(e)}
        _log_order_result('place_buy', 'buy', order, resp)
        return resp


def fetch_alpaca_account_and_positions():
    """Fetch Alpaca account info and positions. Returns (acct_dict, positions_list) or (None, None) on failure."""
    key = os.getenv('ALPACA_KEY')
    secret = os.getenv('ALPACA_SECRET')
    if not key or not secret:
        return None, None
    base = 'https://paper-api.alpaca.markets' if env_bool('ALPACA_PAPER', True) else 'https://api.alpaca.markets'
    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}
    try:
        r = requests.get(f"{base}/v2/account", headers=headers, timeout=10)
        r.raise_for_status()
        acct = r.json()
    except Exception:
        return None, None
    try:
        rpos = requests.get(f"{base}/v2/positions", headers=headers, timeout=10)
        rpos.raise_for_status()
        positions = rpos.json()
    except Exception:
        positions = []
    return acct, positions


from project_config import SAFE_ORDERS_LOG, ORDER_AUDIT, MAG7_ANALYSIS, MAG7_REALLOC_PLAN


def _append_order_audit(entry: dict):
    """Append a JSON line to ./data/order_audit.jsonl for auditing placed/attempted orders."""
    try:
        with open(ORDER_AUDIT, 'a') as jf:
            jf.write(json.dumps(entry) + '\n')
    except Exception:
        pass


def _normalize_order_resp(resp):
    """Return a JSON-serializable representation of an order response."""
    if isinstance(resp, requests.Response):
        try:
            data = resp.json()
        except ValueError:
            data = resp.text
        return {'status_code': resp.status_code, 'body': data}
    return resp


def _json_default(obj):
    if isinstance(obj, requests.Response):
        return _normalize_order_resp(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

def _round_limit_price(price: float | None) -> float | None:
    if price is None:
        return None
    decimals = 4 if price < 1 else 2
    try:
        return round(float(price), decimals)
    except Exception:
        return None


def _get_extended_hours_limit_pct() -> float:
    try:
        pct = _env_float('EXTENDED_HOURS_LIMIT_PCT', 0.01)
    except Exception:
        pct = 0.01
    if pct < 0:
        pct = abs(pct)
    return min(pct, 0.2)


def _is_regular_market_hours(now_utc: datetime | None = None) -> bool:
    now_utc = now_utc or datetime.now(timezone.utc)
    et = now_utc.astimezone(zoneinfo.ZoneInfo('America/New_York'))
    if et.weekday() >= 5:
        return False
    if is_market_holiday(et):
        return False
    start = time(9, 30)
    end = time(16, 0)
    return start <= et.time() <= end


def _should_use_extended_hours(enabled: bool, now_utc: datetime | None = None) -> bool:
    if not enabled:
        return False
    if env_bool('EXTENDED_HOURS_ALWAYS', False):
        return True
    return not _is_regular_market_hours(now_utc)


def _limit_price_for_extended_hours(reference_price: float | None, side: str) -> float | None:
    if reference_price is None or reference_price <= 0:
        return None
    side = (side or '').lower()
    if side not in ('buy', 'sell'):
        return None
    offset = _get_extended_hours_limit_pct()
    if side == 'buy':
        price = reference_price * (1 + offset)
    else:
        price = reference_price * (1 - offset)
    return _round_limit_price(price)


def _maybe_add_extended_hours(order: dict, enabled: bool, reference_price: float | None = None, now_utc: datetime | None = None):
    if not _should_use_extended_hours(enabled, now_utc):
        return
    if order.get('notional') is not None:
        return
    # Alpaca only supports extended-hours on limit orders.
    if order.get('type') == 'market':
        limit_price = _limit_price_for_extended_hours(reference_price, order.get('side'))
        if limit_price is None:
            logger.info('extended_hours enabled but missing reference price; leaving market order for %s', order.get('symbol'))
            return
        order['type'] = 'limit'
        order['limit_price'] = limit_price
    if order.get('type') == 'limit':
        order['extended_hours'] = True
        order.setdefault('time_in_force', 'day')


def _log_order_result(flow: str, action: str, payload: dict, resp):
    """Emit human-friendly log entries for order attempts."""
    resp = _normalize_order_resp(resp)
    symbol = payload.get('symbol') if isinstance(payload, dict) else payload
    qty = payload.get('qty') if isinstance(payload, dict) else None
    if isinstance(resp, dict):
        order_id = resp.get('id') or resp.get('client_order_id')
        status = resp.get('status') or resp.get('error') or 'unknown'
        if resp.get('error'):
            logger.error("%s %s order for %s qty=%s failed: %s", flow, action, symbol, qty, resp.get('error'))
        else:
            logger.info("%s %s order for %s qty=%s submitted (id=%s status=%s)", flow, action, symbol, qty, order_id, status)
    else:
        logger.info("%s %s order for %s qty=%s response: %s", flow, action, symbol, qty, resp)


def _parse_iso8601(ts: str):
    if not ts:
        return None
    try:
        if ts.endswith('Z'):
            ts = ts[:-1] + '+00:00'
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _summarize_order_metrics(entries: list) -> list:
    metrics = []
    for entry in entries:
        resp = entry.get('resp') if isinstance(entry, dict) else {}
        if not isinstance(resp, dict):
            resp = {}
        submitted = _parse_iso8601(resp.get('submitted_at'))
        filled = _parse_iso8601(resp.get('filled_at'))
        fill_time = None
        if submitted and filled:
            try:
                fill_time = max(0.0, (filled - submitted).total_seconds())
            except Exception:
                fill_time = None
        filled_avg_price = None
        fprice = resp.get('filled_avg_price')
        if fprice is not None:
            try:
                filled_avg_price = float(fprice)
            except Exception:
                filled_avg_price = None
        ref_price = entry.get('reference_price')
        slippage = None
        if filled_avg_price is not None and ref_price is not None:
            try:
                slippage = filled_avg_price - float(ref_price)
            except Exception:
                slippage = None
        metrics.append({
            'symbol': entry.get('symbol'),
            'action': entry.get('action'),
            'qty': entry.get('qty'),
            'status': resp.get('status') or resp.get('error'),
            'fill_time_seconds': fill_time,
            'filled_avg_price': filled_avg_price,
            'reference_price': ref_price,
            'slippage': slippage,
        })
    return metrics


def _post_trade_validation(top_syms: list, target_value: float, total_portfolio_value: float):
    acct, positions = fetch_alpaca_account_and_positions()
    acct = acct or {}
    positions = positions or []
    pos_map = {p.get('symbol'): p for p in positions if isinstance(p, dict)}
    results = []
    for sym in top_syms:
        pos = pos_map.pop(sym, {})
        actual_val = 0.0
        actual_qty = 0.0
        try:
            actual_val = float(pos.get('market_value') or 0.0)
        except Exception:
            actual_val = 0.0
        try:
            actual_qty = float(pos.get('qty') or 0.0)
        except Exception:
            actual_qty = 0.0
        results.append({
            'symbol': sym,
            'target_value': target_value,
            'actual_value': actual_val,
            'delta_value': actual_val - target_value,
            'actual_qty': actual_qty,
        })
    if pos_map:
        other_value = 0.0
        other_qty = 0.0
        for pos in pos_map.values():
            try:
                other_value += float(pos.get('market_value') or 0.0)
            except Exception:
                pass
            try:
                other_qty += float(pos.get('qty') or 0.0)
            except Exception:
                pass
        results.append({'symbol': 'OTHER', 'target_value': 0.0, 'actual_value': other_value, 'delta_value': other_value, 'actual_qty': other_qty})
    cash_target = None
    try:
        cash_target = max(0.0, total_portfolio_value - (target_value * max(1, len(top_syms))))
    except Exception:
        cash_target = None
    try:
        current_cash = float(acct.get('cash') or 0.0)
    except Exception:
        current_cash = None
    cash_delta = None
    if cash_target is not None and current_cash is not None:
        cash_delta = current_cash - cash_target
    return {
        'positions': results,
        'cash': {'current': current_cash, 'target': cash_target, 'delta': cash_delta},
    }


def close_all_positions():
    key = os.getenv('ALPACA_KEY')
    secret = os.getenv('ALPACA_SECRET')
    if not key or not secret:
        return {'error': 'missing alpaca keys'}
    base = 'https://paper-api.alpaca.markets' if env_bool('ALPACA_PAPER', True) else 'https://api.alpaca.markets'
    headers = {
        'APCA-API-KEY-ID': key,
        'APCA-API-SECRET-KEY': secret,
        'Content-Type': 'application/json',
    }
    extended_hours_enabled = env_bool('EXTENDED_HOURS', False)
    res = {'closed': [], 'errors': []}
    try:
        r = requests.get(f"{base}/v2/positions", headers=headers, timeout=10)
        r.raise_for_status()
        positions = r.json()
    except Exception as e:
        return {'error': f'failed to list positions: {e}'}

    for p in positions:
        sym = p.get('symbol')
        qty = int(float(p.get('qty', 0)))
        if qty <= 0:
            continue
        ref_price = None
        try:
            ref_price = float(p.get('current_price') or 0.0)
        except Exception:
            ref_price = None
        if not ref_price:
            try:
                mv = float(p.get('market_value') or 0.0)
                ref_price = mv / qty if qty else None
            except Exception:
                ref_price = None
        order = {'symbol': sym, 'qty': qty, 'side': 'sell', 'type': 'market', 'time_in_force': 'day'}
        _maybe_add_extended_hours(order, extended_hours_enabled, reference_price=ref_price)
        try:
            orr = requests.post(f"{base}/v2/orders", headers=headers, json=order, timeout=10)
            orr.raise_for_status()
            res['closed'].append({'symbol': sym, 'qty': qty, 'resp': orr.json()})
            _log_order_result('close_all_positions', 'sell', order, res['closed'][-1]['resp'])
        except Exception as e:
            res['errors'].append({'symbol': sym, 'error': str(e)})
            _log_order_result('close_all_positions', 'sell', order, {'error': str(e)})
    return res


def invest_flow():
    symbols = get_symbols()
    candidates = analyze_and_choose(symbols)
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'candidates': candidates}
    summary_guard = out.setdefault('summary', {}).setdefault('guard', {'failed': [], 'sector_skipped': []})
    # fetch current account and positions to avoid double-buying
    acct, positions = fetch_alpaca_account_and_positions()
    out['account'] = acct
    out['positions'] = positions
    try:
        import eod_fetcher as _ef
        out['openai_single_calls'] = getattr(_ef, '_OPENAI_SINGLE_CALLS', 0)
        out['openai_batch_calls'] = getattr(_ef, '_OPENAI_BATCH_CALLS', 0)
    except Exception:
        pass
    # pick top-N candidates that recommend buy
    top_n = _env_int('INVEST_TOP_N', 1)
    invest_total_pct = _env_float('INVEST_TOTAL_PCT', 0.01)
    rebalance = env_bool('REBALANCE', False)

    buys = [c for c in candidates if c['suggested'] == 'buy']
    # filter out symbols already held unless rebalance is enabled
    held_symbols = set([p.get('symbol') for p in (positions or [])])
    if not rebalance and held_symbols:
        buys = [c for c in buys if c['symbol'] not in held_symbols]
    if not buys:
        out['note'] = 'no buy recommendations found'
        try:
            with open(MAG7_ANALYSIS, 'w') as jf:
                json.dump(out, jf, indent=2)
        except Exception:
            pass
        try:
            send_trade_alert('invest_flow', out)
        except Exception as e:
            logger.warning('Trade alert failed: %s', e)
        return out

    # normalize scores 0..1 across today's candidates then scale to 1..100
    scores = [c['score'] for c in buys]
    if scores:
        lo = min(scores)
        hi = max(scores)
    else:
        lo = hi = 0
    def norm(v):
        if hi == lo:
            return 0.5
        return (v - lo) / (hi - lo)

    for c in buys:
        c['norm'] = norm(c['score'])
        c['scaled_score'] = int(max(1, min(100, round(c['norm'] * 99 + 1))))

    # sector guard and selection: pick top_n by scaled_score but limit per-sector to 2
    sorted_buys = sorted(buys, key=lambda x: x['scaled_score'], reverse=True)
    targets = []
    sector_count = {}
    guard_failures = summary_guard['failed']
    sector_skips = summary_guard['sector_skipped']
    for cand in sorted_buys:
        if len(targets) >= top_n:
            break
        # get sector from premarket if available
        prem = cand.get('analysis') if isinstance(cand.get('analysis'), dict) else None
        sector = None
        if prem and isinstance(prem, dict):
            sector = prem.get('premarket', {}).get('sector')
        if not sector:
            try:
                import eod_fetcher as _ef
                sector = _ef.fetch_premarket_info(cand['symbol']).get('sector')
            except Exception:
                sector = None
        cand['sector'] = sector or 'UNKNOWN'

        # enforce guards: price >= 3, premarket dollar vol >= 2M, spread <= 0.5%
        preminfo = None
        try:
            import eod_fetcher as _ef
            preminfo = _ef.fetch_premarket_info(cand['symbol'])
        except Exception:
            preminfo = {}
        price_ok = (cand.get('latest_price') or 0) >= 3
        dv = preminfo.get('dollar_volume') or 0
        vol_ok = (dv and float(dv) >= 2_000_000)
        spread_ok = (preminfo.get('spread') is None) or (float(preminfo.get('spread') or 0) <= 0.005)
        if not (price_ok and vol_ok and spread_ok):
            cand['guard_failed'] = True
            guard_failures.append({
                'symbol': cand['symbol'],
                'price_ok': bool(price_ok),
                'volume_ok': bool(vol_ok),
                'spread_ok': bool(spread_ok),
            })
            continue
        # sector cap: at most 2 per sector
        if sector_count.get(cand['sector'], 0) >= 2:
            sector_skips.append({'symbol': cand['symbol'], 'sector': cand['sector']})
            continue
        sector_count[cand['sector']] = sector_count.get(cand['sector'], 0) + 1
        cand['guard_failed'] = False
        targets.append(cand)

    # compute qtys per target based on invest_total_pct of buying power
    key = os.getenv('ALPACA_KEY')
    secret = os.getenv('ALPACA_SECRET')
    if not key or not secret:
        out['error'] = 'missing alpaca keys'
        try:
            with open(MAG7_ANALYSIS, 'w') as jf:
                json.dump(out, jf, indent=2)
        except Exception:
            pass
        try:
            send_trade_alert('invest_flow', out)
        except Exception as e:
            logger.warning('Trade alert failed: %s', e)
        return out

    base = 'https://paper-api.alpaca.markets' if env_bool('ALPACA_PAPER', True) else 'https://api.alpaca.markets'
    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}
    try:
        r = requests.get(f"{base}/v2/account", headers=headers, timeout=10)
        r.raise_for_status()
        acct = r.json()
        buying_power = float(acct.get('buying_power') or acct.get('cash') or 0.0)
    except Exception as e:
        out['error'] = f'failed to fetch account: {e}'
        try:
            with open(MAG7_ANALYSIS, 'w') as jf:
                json.dump(out, jf, indent=2)
        except Exception:
            pass
        try:
            send_trade_alert('invest_flow', out)
        except Exception as ex:
            logger.warning('Trade alert failed: %s', ex)
        return out

    pct = _env_float('INVEST_SIZE_PCT', 0.01)
    target_usd = buying_power * pct
    qtys = []
    per_target_usd = target_usd * (invest_total_pct / pct) if pct > 0 else target_usd
    per_target_usd = (buying_power * invest_total_pct) / max(1, len(targets))
    for t in targets:
        latest_price = float(t['latest_price'])
        q = int(math.floor(per_target_usd / latest_price))
        qtys.append({'symbol': t['symbol'], 'qty': q, 'price': latest_price, 'confidence': t['confidence']})

    out['targets'] = qtys

    if all(q['qty'] < 1 for q in qtys):
        out['error'] = 'calculated qty < 1 for all targets'
        try:
            with open(MAG7_ANALYSIS, 'w') as jf:
                json.dump(out, jf, indent=2)
        except Exception:
            pass
        try:
            send_trade_alert('invest_flow', out)
        except Exception as e:
            logger.warning('Trade alert failed: %s', e)
        return out

    if not env_bool('PLACE_ORDER', False):
        out['note'] = 'PLACE_ORDER not enabled; dry run'
        try:
            with open(MAG7_ANALYSIS, 'w') as jf:
                json.dump(out, jf, indent=2)
        except Exception:
            pass
        logger.info('invest_flow: PLACE_ORDER disabled; no buy orders were submitted')
        try:
            send_trade_alert('invest_flow', out)
        except Exception as e:
            logger.warning('Trade alert failed: %s', e)
        return out

    # optionally rebalance: sell positions not in targets
    if rebalance:
        try:
            rpos = requests.get(f"{base}/v2/positions", headers=headers, timeout=10)
            rpos.raise_for_status()
            positions = rpos.json()
            existing_syms = [p.get('symbol') for p in positions]
            target_syms = [t['symbol'] for t in qtys]
            for sym in existing_syms:
                if sym not in target_syms:
                    # sell full position
                    pos = [p for p in positions if p.get('symbol') == sym][0]
                    pqty = int(float(pos.get('qty', 0)))
                    if pqty > 0:
                        ref_price = None
                        try:
                            ref_price = float(pos.get('current_price') or 0.0)
                        except Exception:
                            ref_price = None
                        if not ref_price:
                            try:
                                mv = float(pos.get('market_value') or 0.0)
                                ref_price = mv / pqty if pqty else None
                            except Exception:
                                ref_price = None
                        order = {'symbol': sym, 'qty': pqty, 'side': 'sell', 'type': 'market', 'time_in_force': 'day'}
                        _maybe_add_extended_hours(order, extended_hours_enabled, reference_price=ref_price)
                        try:
                            requests.post(f"{base}/v2/orders", headers=headers, json=order, timeout=10)
                        except Exception:
                            pass
        except Exception:
            pass

    # interactive/forced confirmation for buys
    if not env_bool('INVEST_YES', False) and not env_bool('INVEST_FORCE', False):
        try:
            ans = input(f"Place buy orders for targets {[(q['symbol'], q['qty']) for q in qtys]}? Type 'yes' to confirm: ").strip().lower()
        except EOFError:
            ans = ''
        if ans != 'yes':
            out['note'] = 'user declined'
            with open('./data/investment_action.json', 'w') as jf:
                json.dump(out, jf, indent=2)
            try:
                send_trade_alert('invest_flow', out)
            except Exception as e:
                logger.warning('Trade alert failed: %s', e)
            return out

    extended_hours_enabled = env_bool('EXTENDED_HOURS', False)
    orders = []
    for q in qtys:
        if q['qty'] < 1:
            continue
        # place idempotent extended-hours order with client_order_id = date+ticker
        coid = f"{datetime.now(timezone.utc).date().isoformat()}::{q['symbol']}"
        try:
            order = {'symbol': q['symbol'], 'qty': q['qty'], 'side': 'buy', 'type': 'market', 'time_in_force': 'day', 'client_order_id': coid}
            _maybe_add_extended_hours(order, extended_hours_enabled, reference_price=q.get('price'))
            r = requests.post(f"{base}/v2/orders", headers=headers, json=order, timeout=10)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            resp = {'error': str(e)}
        norm_resp = _normalize_order_resp(resp)
        orders.append({'symbol': q['symbol'], 'qty': q['qty'], 'action': 'buy', 'reference_price': q.get('price'), 'resp': norm_resp})
        # audit the attempted order
        _append_order_audit({'ts': datetime.now(timezone.utc).isoformat(), 'flow': 'invest_flow', 'symbol': q['symbol'], 'qty': q['qty'], 'client_order_id': coid, 'resp': norm_resp})
        _log_order_result('invest_flow', 'buy', order, norm_resp)
    out['orders'] = orders
    out.setdefault('summary', {})['orders'] = _summarize_order_metrics(orders)
    with open('./data/investment_action.json', 'w') as jf:
        json.dump(out, jf, indent=2)
    try:
        send_trade_alert('invest_flow', out)
    except Exception as e:
        logger.warning('Trade alert failed: %s', e)
    return out


def analyze_mag7_and_invest(reallocate_full_arg=None, place_order_arg=None, invest_yes_arg=None, order_executor=None):
    """Analyze the Mag Seven universe and equally invest in the top-ranked trio.

    Behavior:
    - Determine candidate symbols from the fixed Mag Seven list (or SYMBOLS env override).
    - For each symbol, fetch EOD (recent) data, Alpaca news (24-36h), and analyst entries.
    - Score candidates locally (technical + news) to get 0-100 composite rankings.
    - Equally allocate INVEST_TOTAL_PCT of buying power across the top three symbols and place orders (if PLACE_ORDER enabled).
    """
    # discover top symbols via Polygon news when available; fallback to env SYMBOLS
    symbols = discover_top_symbols_via_news(top_n=10)
    # gather data
    items = []
    t0 = datetime.now().timestamp()
    data = fetch_eod(symbols, output_dir='./data')
    _tprint('fetch_eod_all', t0)
    # per-symbol news/analyst fetch (parallelized)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch_for_symbol(sym: str):
        df = data.get(sym)
        latest_price = None
        if df is not None and not df.empty:
            try:
                latest_price = float(df.iloc[-1]['Close'])
            except Exception:
                latest_price = None
        intraday_df = fetch_alpaca_intraday_bars(sym)
        indicators = _compute_indicator_snapshot(df, intraday_df)
        t_sym = datetime.now().timestamp()
        use_web = env_bool('USE_WEB_RESEARCH', False)
        if use_web:
            articles, agg_sent = [], 0.0
        else:
            articles, agg_sent = aggregate_news_for_symbol(sym, limit=20, hours=24)
        try:
            import eod_fetcher as _ef
            poly = _ef.fetch_polygon_analyst_ratings(sym)
            analyst_entries = poly.get('entries') if isinstance(poly, dict) else []
            analyst_sent = poly
        except Exception:
            analyst_entries = []
            analyst_sent = {}
        _tprint(f'news+analyst:{sym}', t_sym)
        news_conf = 0.0
        if articles:
            confidences = [float(a.get('gpt_confidence') or a.get('confidence') or 0.0) for a in articles]
            if confidences:
                news_conf = sum(confidences) / len(confidences)
        top_catalysts = _collect_news_catalysts(articles)
        metrics = {'close': latest_price, 'news_aggregate_sentiment': agg_sent, 'news_articles': articles, 'analyst_entries': analyst_entries, 'analyst_sentiment': analyst_sent}
        metrics['indicators'] = indicators
        metrics['latest_price'] = indicators.get('close') or latest_price
        metrics['news_confidence'] = news_conf
        metrics['top_catalysts'] = top_catalysts
        return {'symbol': sym, 'metrics': metrics, 'news_articles': articles, 'analyst_entries': analyst_entries}

    t_symbols_start = datetime.now().timestamp()
    workers = _env_int('FETCH_WORKERS', 6)
    items = []
    # use a bounded thread pool to parallelize I/O-bound fetches
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_for_symbol, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                res = fut.result()
                items.append(res)
            except Exception:
                # keep going if an individual symbol fails
                pass
    _tprint('per_symbol_fetches', t_symbols_start)

    t_score = datetime.now().timestamp()
    local_scores = _score_items(items)
    scores = local_scores
    try:
        model_scores = analyze_with_chatgpt.analyze_portfolio_short_term(items)
        if model_scores:
            local_map = {s.get('symbol'): s for s in local_scores if s.get('symbol')}
            merged = []
            for entry in model_scores:
                sym = entry.get('symbol')
                if not sym:
                    continue
                merged_entry = dict(entry)
                base = local_map.get(sym)
                if base:
                    for key in ('details', 'rationale_bullets', 'confidence', 'decision', 'score_raw'):
                        if key in base and key not in merged_entry:
                            merged_entry[key] = base[key]
                merged.append(merged_entry)
            if merged:
                merged.sort(key=lambda x: x.get('score', 0), reverse=True)
                for idx, entry in enumerate(merged, 1):
                    entry['rank'] = idx
                scores = merged
    except Exception:
        scores = local_scores
    _tprint('scoring', t_score)

    # attach detailed summaries for each symbol so downstream outputs include human-readable context
    analysis_map: dict[str, dict] = {}
    for item in items:
        sym = item.get('symbol')
        metrics = item.get('metrics') or {}
        if not sym or not metrics:
            continue
        try:
            analysis_map[sym] = analyze_summary(sym, metrics)
        except Exception as exc:
            analysis_map[sym] = {
                'summary': f'analysis failed: {exc}',
                'signals': {},
                'suggested_action': None,
                'confidence': 0.0,
                'recommended_size': None,
                'rationale': None,
            }

    for entry in scores:
        sym = entry.get('symbol')
        if sym in analysis_map:
            entry['analysis'] = analysis_map[sym]

    # pick top N for allocation
    try:
        allocate_count = _env_int('TOP_ALLOCATE_COUNT', DEFAULT_TOP_ALLOCATE_COUNT)
    except Exception:
        allocate_count = DEFAULT_TOP_ALLOCATE_COUNT
    allocate_count = max(1, min(allocate_count, len(scores)))
    top_candidates = []
    for x in scores[:allocate_count]:
        top_candidates.append(dict(x))

    # allocate equally INVEST_TOTAL_PCT of buying power across selected names
    invest_total_pct = _env_float('INVEST_TOTAL_PCT', 0.01)
    key = os.getenv('ALPACA_KEY')
    secret = os.getenv('ALPACA_SECRET')
    # fetch current account & positions
    acct, positions = fetch_alpaca_account_and_positions()
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'scores': scores, 'top': top_candidates, 'account': acct, 'positions': positions}
    summary_guard = out.setdefault('summary', {}).setdefault('guard', {'failed': [], 'sector_skipped': []})
    if not key or not secret:
        out['error'] = 'missing alpaca keys'
        try:
            send_trade_alert('analyze_mag7_and_invest', out)
        except Exception as e:
            logger.warning('Trade alert failed: %s', e)
        return out
    if not top_candidates:
        out['error'] = 'no scoring results produced; skipping allocation changes'
        try:
            send_trade_alert('analyze_mag7_and_invest', out)
        except Exception as e:
            logger.warning('Trade alert failed: %s', e)
        return out
    base = 'https://paper-api.alpaca.markets' if env_bool('ALPACA_PAPER', True) else 'https://api.alpaca.markets'
    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}
    place_orders = place_order_arg if place_order_arg is not None else env_bool('PLACE_ORDER', False)

    buying_power = None
    if isinstance(acct, dict):
        bp_val = acct.get('buying_power') or acct.get('cash')
        try:
            if bp_val is not None:
                buying_power = float(bp_val)
        except Exception:
            buying_power = None
    if buying_power is None:
        try:
            r = requests.get(f"{base}/v2/account", headers=headers, timeout=10)
            r.raise_for_status()
            acct = r.json()
            buying_power = float(acct.get('buying_power') or acct.get('cash') or 0.0)
        except Exception as e:
            out['error'] = f'failed to fetch account: {e}'
            try:
                send_trade_alert('analyze_mag7_and_invest', out)
            except Exception as ex:
                logger.warning('Trade alert failed: %s', ex)
            return out

    per_target_usd = (buying_power * invest_total_pct) / max(1, len(top_candidates))
    # Reallocation mode: if REALLOCATE_FULL is set, compute portfolio value (cash + positions)
    # resolve reallocate flag via explicit arg (if provided) or env var
    reallocate_full = reallocate_full_arg if reallocate_full_arg is not None else env_bool('REALLOCATE_FULL', False)
    held_symbols = set([p.get('symbol') for p in (positions or [])])
    extended_hours_enabled = env_bool('EXTENDED_HOURS', False)

    if reallocate_full:
        # compute total portfolio value using latest prices from `data` and account cash
        cash = float(acct.get('cash') or 0.0) if isinstance(acct, dict) else 0.0
        pv = cash
        pos_map = {}
        for p in (positions or []):
            sym = p.get('symbol')
            qty = float(p.get('qty') or 0)
            price = None
            dfp = data.get(sym)
            if dfp is not None and not dfp.empty:
                try:
                    price = float(dfp.iloc[-1]['Close'])
                except Exception:
                    price = None
            market_val = (price or 0.0) * qty
            pv += market_val
            pos_map[sym] = {'qty': qty, 'price': price, 'market_value': market_val}

        # target per top symbol (equal split of total portfolio across selected symbols)
        top_syms = [t.get('symbol') for t in top_candidates]
        target_per_symbol = pv / max(1, len(top_syms))

        sells = []
        buys = []
        # plan sells: for any held symbol not in top_syms, plan full sell
        for p in (positions or []):
            sym = p.get('symbol')
            qty = int(float(p.get('qty') or 0))
            if qty <= 0:
                continue
            if sym not in top_syms:
                sells.append({'symbol': sym, 'qty': qty})

        # plan target buys to reach target allocation for each top symbol
        for t in top_candidates:
            sym = t.get('symbol')
            cur_qty = int(float(pos_map.get(sym, {}).get('qty', 0)))
            cur_price = float(pos_map.get(sym, {}).get('price') or 0.0)
            cur_value = float(pos_map.get(sym, {}).get('market_value') or 0.0)
            needed_usd = target_per_symbol - cur_value
            if needed_usd <= 0:
                # already at or above target
                continue
            price = None
            dfp = data.get(sym)
            if dfp is not None and not dfp.empty:
                try:
                    price = float(dfp.iloc[-1]['Close'])
                except Exception:
                    price = None
            if not price or price <= 0:
                continue
            buy_qty = int(math.floor(needed_usd / price))
            if buy_qty > 0:
                buys.append({'symbol': sym, 'qty': buy_qty, 'price': price, 'needed_usd': needed_usd})

        out['reallocation_plan'] = {'portfolio_value': pv, 'cash': cash, 'target_per_symbol': target_per_symbol, 'sells': sells, 'buys': buys}
        # if PLACE_ORDER not enabled, return dry-run plan
        if not place_orders:
            try:
                with open(MAG7_REALLOC_PLAN, 'w') as tf:
                    json.dump(out, tf, indent=2)
            except Exception:
                pass
            logger.info('reallocate_full: PLACE_ORDER disabled; generated plan only (no trades)')
            try:
                send_trade_alert('reallocate_full', out)
            except Exception as e:
                logger.warning('Trade alert failed: %s', e)
            return out

        # If PLACE_ORDER is enabled, place sells first then buys and audit them
        # Note: keep implementation simple - place market sells for sells, then buys for buys
        placed = []
        # executor helper: use injected order_executor when provided (use requests.post-like default)
        if order_executor is None:
            def _exec(url, headers=None, json=None, timeout=10):
                return requests.post(url, headers=headers, json=json, timeout=timeout)
        else:
            _exec = order_executor

        for s in sells:
            resp = None
            ref_price = None
            base_pos = pos_map.get(s['symbol']) if 'pos_map' in locals() else None
            try:
                if base_pos and s['qty']:
                    ref_price = float(base_pos.get('price') or 0.0) or None
                    if not ref_price:
                        mv = float(base_pos.get('market_value') or 0.0)
                        qty = float(s['qty']) or 1.0
                        ref_price = mv / qty if qty else None
            except Exception:
                ref_price = None
            try:
                order = {'symbol': s['symbol'], 'qty': s['qty'], 'side': 'sell', 'type': 'market', 'time_in_force': 'day'}
                _maybe_add_extended_hours(order, extended_hours_enabled, reference_price=ref_price)
                r = _exec(f"{base}/v2/orders", headers=headers, json=order, timeout=10)
                if isinstance(r, dict):
                    resp = r
                else:
                    r.raise_for_status()
                    resp = r.json()
            except Exception as e:
                resp = {'error': str(e)}
            norm_resp = _normalize_order_resp(resp)
            placed.append({'action': 'sell', 'symbol': s['symbol'], 'qty': s['qty'], 'reference_price': ref_price, 'resp': norm_resp})
            _append_order_audit({'ts': datetime.now(timezone.utc).isoformat(), 'flow': 'reallocate_full', 'action': 'sell', 'symbol': s['symbol'], 'qty': s['qty'], 'resp': norm_resp})
            _log_order_result('reallocate_full', 'sell', {'symbol': s['symbol'], 'qty': s['qty']}, norm_resp)

        for b in buys:
            sym = b['symbol']
            coid = f"{datetime.now(timezone.utc).date().isoformat()}::{sym}"
            resp = None
            # first attempt: place by qty
            payload_qty = {'symbol': sym, 'qty': b['qty'], 'side': 'buy', 'type': 'market', 'time_in_force': 'day', 'client_order_id': coid}
            _maybe_add_extended_hours(payload_qty, extended_hours_enabled, reference_price=b.get('price'))
            try:
                r = _exec(f"{base}/v2/orders", headers=headers, json=payload_qty, timeout=10)
                if isinstance(r, dict):
                    resp = r
                else:
                    r.raise_for_status()
                    resp = r.json()
            except Exception as he:
                status = None
                body = None
                try:
                    status = getattr(he, 'response', None) and he.response.status_code
                    body = getattr(he, 'response', None) and he.response.text
                except Exception:
                    body = str(he)
                # If 422 Unprocessable Entity, attempt a notional-based order (fractional) as a fallback
                allow_notional_fallback = payload_qty.get('type') == 'market' and not _should_use_extended_hours(extended_hours_enabled)
                if status == 422 and allow_notional_fallback:
                    try:
                        notional = str(round(float(b.get('needed_usd') or 0.0), 2))
                        payload_notional = {'symbol': sym, 'notional': notional, 'side': 'buy', 'type': 'market', 'time_in_force': 'day', 'client_order_id': coid}
                        _maybe_add_extended_hours(payload_notional, extended_hours_enabled, reference_price=b.get('price'))
                        import time

                        time.sleep(0.5)
                        r2 = _exec(f"{base}/v2/orders", headers=headers, json=payload_notional, timeout=10)
                        if isinstance(r2, dict):
                            resp = r2
                        else:
                            r2.raise_for_status()
                            resp = r2.json()
                    except Exception as e2:
                        resp = {'error': str(e2), 'status': status, 'body': body}
                else:
                    resp = {'error': str(he), 'status': status, 'body': body}
            except Exception as e:
                resp = {'error': str(e)}

            norm_resp = _normalize_order_resp(resp)
            placed.append({'action': 'buy', 'symbol': sym, 'qty': b['qty'], 'reference_price': b.get('price'), 'resp': norm_resp})
            # audit with payloads and response
            try:
                audit_entry = {'ts': datetime.now(timezone.utc).isoformat(), 'flow': 'reallocate_full', 'action': 'buy', 'symbol': sym, 'qty': b['qty'], 'client_order_id': coid, 'resp': norm_resp}
                if resp and isinstance(resp, dict) and resp.get('error') and 'notional' in str(resp.get('body', '')):
                    audit_entry['attempted_notional'] = True
                _append_order_audit(audit_entry)
            except Exception:
                pass
            _log_order_result('reallocate_full', 'buy', {'symbol': sym, 'qty': b['qty']}, norm_resp)

        order_summary = _summarize_order_metrics(placed)
        post_trade = _post_trade_validation(top_syms, target_per_symbol, pv)
        out['placed'] = placed
        out.setdefault('summary', {})['orders'] = order_summary
        out['summary']['post_trade'] = post_trade
        try:
            send_trade_alert('reallocate_full', out)
        except Exception as e:
            logger.warning('Trade alert failed: %s', e)
        return out

    # not reallocate_full: existing equal-invest flow
    to_buy = top_candidates
    if not env_bool('REBALANCE', False) and held_symbols:
        to_buy = [t for t in top_candidates if t.get('symbol') not in held_symbols]

    orders = []
    for t in to_buy:
        sym = t.get('symbol')
        # find latest price from data
        df = data.get(sym)
        if df is None or df.empty:
            continue
        price = float(df.iloc[-1]['Close'])
        qty = int(math.floor(per_target_usd / price))
        orders.append({'symbol': sym, 'qty': qty, 'price': price, 'score': t.get('score')})

    out['planned'] = orders
    if not place_orders:
        out['note'] = 'PLACE_ORDER not enabled; dry run'
        # export planned Mag-7 analysis for review
        try:
            os.makedirs('./data', exist_ok=True)
            with open('./data/mag7_analysis.json', 'w') as tf:
                json.dump(out, tf, indent=2)
        except Exception:
            pass
        logger.info('analyze_mag7_and_invest: PLACE_ORDER disabled; generated allocation plan without trading')
        try:
            send_trade_alert('analyze_mag7_and_invest', out)
        except Exception as e:
            logger.warning('Trade alert failed: %s', e)
        return out

    # place orders
    placed = []
    for o in orders:
        if o['qty'] < 1:
            placed.append({'symbol': o['symbol'], 'qty': o['qty'], 'error': 'qty<1'})
            continue
        try:
            payload = {'symbol': o['symbol'], 'qty': o['qty'], 'side': 'buy', 'type': 'market', 'time_in_force': 'day'}
            _maybe_add_extended_hours(payload, extended_hours_enabled, reference_price=o.get('price'))
            r = requests.post(f"{base}/v2/orders", headers=headers, json=payload, timeout=10)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            resp = {'error': str(e)}
        norm_resp = _normalize_order_resp(resp)
        placed.append({'symbol': o['symbol'], 'qty': o['qty'], 'action': 'buy', 'reference_price': o.get('price'), 'resp': norm_resp})
        # audit placed order
        _append_order_audit({'ts': datetime.now(timezone.utc).isoformat(), 'flow': 'analyze_mag7_and_invest', 'symbol': o['symbol'], 'qty': o['qty'], 'resp': norm_resp})
        payload = {'symbol': o['symbol'], 'qty': o['qty']}
        _log_order_result('analyze_mag7_and_invest', 'buy', payload, norm_resp)

    order_summary = _summarize_order_metrics(placed)
    out['placed'] = placed
    out.setdefault('summary', {})['orders'] = order_summary
    try:
        send_trade_alert('analyze_mag7_and_invest', out)
    except Exception as e:
        logger.warning('Trade alert failed: %s', e)
    return out


analyze_top10_and_invest = analyze_mag7_and_invest


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--invest', action='store_true')
    parser.add_argument('--mag7', action='store_true', dest='mag7', help='Run Mag-7 analysis and optionally place equal allocations to the top three')
    parser.add_argument('--top10', action='store_true', dest='mag7', help=argparse.SUPPRESS)
    parser.add_argument('--close', action='store_true')
    args = parser.parse_args()

    os.makedirs('./data', exist_ok=True)

    if args.invest:
        out = invest_flow()
        print(json.dumps(out, indent=2, default=_json_default))
    elif args.mag7:
        out = analyze_mag7_and_invest()
        # export results
        try:
            with open(MAG7_ANALYSIS, 'w') as tf:
                json.dump(out, tf, indent=2)
        except Exception:
            pass
        print(json.dumps(out, indent=2, default=_json_default))
    elif args.close:
        res = close_all_positions()
        if res.get('closed'):
            try:
                send_trade_alert('close_all', res)
            except Exception as e:
                logger.warning('Trade alert failed: %s', e)
        try:
            with open(MAG7_ANALYSIS.replace('mag7_analysis.json', 'close_action.json'), 'w') as jf:
                json.dump(res, jf, indent=2)
        except Exception:
            pass
        print(json.dumps(res, indent=2, default=_json_default))
    else:
        print('No action specified; use --invest or --close')


if __name__ == '__main__':
    main()
