import os
import time
import math
import logging
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf

import requests
from datetime import datetime, timedelta, timezone
import hashlib
import json
from urllib.parse import urlparse
from collections import Counter
from dotenv import load_dotenv
from openai_helpers import call_chat_completion, _is_gpt5_model

load_dotenv()

# local debug helper (avoid importing scheduled_trader)
def debug_enabled():
    return os.getenv('DEBUG', '').strip().lower() in ('1', 'true', 'yes', 'y')

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

from project_config import CACHE_DIR, GPT_ARTICLE_CACHE, SENTIMENT_CACHE, POLY_CACHE

# Configure yfinance caches inside the project tree to avoid permission issues (cron, sandbox, etc.).
_YF_CACHE_ROOT = os.path.join(CACHE_DIR, 'yfinance')
os.makedirs(_YF_CACHE_ROOT, exist_ok=True)
try:
    yf.set_tz_cache_location(os.path.join(_YF_CACHE_ROOT, 'tz'))
except Exception:
    # best-effort; yfinance falls back to defaults if unavailable
    pass

# Quiet noisy yfinance loggers so transient DNS failures do not spam ERROR lines in our logs.
for _logger_name in ('yfinance', 'yfinance.scrapers', 'yfinance.scrapers.quote', 'yfinance.data'):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

# simple disk cache for sentiment results with TTL + max-size (LRU evict)
_SENTIMENT_CACHE_PATH = SENTIMENT_CACHE
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from datetime import time as _time
os.makedirs(CACHE_DIR, exist_ok=True)
try:
    with open(_SENTIMENT_CACHE_PATH, 'r') as _f:
        _SENTIMENT_CACHE = json.load(_f)
except Exception:
    _SENTIMENT_CACHE = {}

# maximum number of cached entries (default 5000)
_SENTIMENT_CACHE_MAX = int(os.getenv('SENTIMENT_CACHE_MAX', 5000))

# Cache entries will be stored as {key: {'score': float, 'ts': epoch_seconds}}
# TTL for cache entries in seconds. Default 30 days.
_SENTIMENT_CACHE_TTL = int(os.getenv('SENTIMENT_CACHE_TTL', 60 * 60 * 24 * 30))
# GPT per-article cache (0..100) path and TTL
_GPT_ARTICLE_CACHE_PATH = GPT_ARTICLE_CACHE
try:
    with open(_GPT_ARTICLE_CACHE_PATH, 'r') as _f:
        _GPT_ARTICLE_CACHE = json.load(_f)
except Exception:
    _GPT_ARTICLE_CACHE = {}

# TTL default 14 days for GPT article cache
_GPT_ARTICLE_CACHE_TTL = int(os.getenv('GPT_ARTICLE_CACHE_TTL', 60 * 60 * 24 * 14))

# Simple counters for telemetry during a run
_OPENAI_SINGLE_CALLS = 0
_OPENAI_BATCH_CALLS = 0


def _http_get(url: str, *, params=None, headers=None, timeout=10, retries=None, raise_for_status=True):
    attempts = retries or int(os.getenv('HTTP_FETCH_RETRIES', 3))
    backoff = float(os.getenv('HTTP_FETCH_BACKOFF', 0.5))
    last_exc = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if raise_for_status:
                resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            time.sleep(backoff * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError('HTTP helper exhausted retries without exception')


def _save_sentiment_cache():
    try:
        # enforce simple LRU by sorting entries by timestamp and trimming
        if isinstance(_SENTIMENT_CACHE, dict) and len(_SENTIMENT_CACHE) > _SENTIMENT_CACHE_MAX:
            # keep newest _SENTIMENT_CACHE_MAX entries
            items = sorted(_SENTIMENT_CACHE.items(), key=lambda kv: int(kv[1].get('ts', 0)), reverse=True)
            trimmed = dict(items[:_SENTIMENT_CACHE_MAX])
        else:
            trimmed = _SENTIMENT_CACHE
        with open(_SENTIMENT_CACHE_PATH, 'w') as _f:
            json.dump(trimmed, _f)
    except Exception:
        pass


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _openai_sentiment(text: str) -> float:
    """Call OpenAI to get sentiment score in [-1,1]. Cache results."""
    if not text:
        return 0.0
    k = _text_key(text[:1000])
    entry = _SENTIMENT_CACHE.get(k)
    model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    if entry is not None:
        cached_model = entry.get('model') if isinstance(entry, dict) else None
        if cached_model and cached_model != model:
            entry = None
        elif cached_model is None and _is_gpt5_model(model):
            entry = None
    if entry is not None:
        try:
            if int(datetime.now(timezone.utc).timestamp()) - int(entry.get('ts', 0)) < _SENTIMENT_CACHE_TTL:
                return float(entry.get('score', 0.0))
            else:
                # expired
                del _SENTIMENT_CACHE[k]
        except Exception:
            pass
        if entry is not None:
            try:
                if int(datetime.now(timezone.utc).timestamp()) - int(entry.get('ts', 0)) < _SENTIMENT_CACHE_TTL:
                    return float(entry.get('score', 0.0))
                else:
                    # expired
                    del _SENTIMENT_CACHE[k]
            except Exception:
                pass
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        score = _simple_sentiment(text)
        _SENTIMENT_CACHE[k] = {
            'score': float(score),
            'ts': int(datetime.now(timezone.utc).timestamp()),
            'model': 'heuristic',
        }
        _save_sentiment_cache()
        return score
    global _OPENAI_SINGLE_CALLS
    # prompt the model to return a numeric sentiment score -1..1
    user_prompt = (
        "Classify the sentiment of the following text about a stock and return ONLY a JSON object with a single key 'score' whose value is a number between -1 and 1.\n\n"
        f"Text: " + text[:1000]
    )
    system_prompt = 'You produce numeric sentiment scores for equity-related text.'
    try:
        txt = call_chat_completion(system_prompt, user_prompt, max_output_tokens=120, temperature=0.0)
        _OPENAI_SINGLE_CALLS += 1
        import re

        m = re.search(r"-?\d+\.?\d*", txt)
        if m:
            val = float(m.group(0))
            val = max(-1.0, min(1.0, val))
        else:
            try:
                j = json.loads(txt)
                val = float(j.get('score', 0.0))
            except Exception:
                val = 0.0
    except Exception:
        val = _simple_sentiment(text)
    # record with timestamp
    _SENTIMENT_CACHE[k] = {
        'score': float(val),
        'ts': int(datetime.now(timezone.utc).timestamp()),
        'model': model,
    }
    _save_sentiment_cache()
    return val


def _save_gpt_article_cache():
    try:
        # trim by timestamp if huge
        if isinstance(_GPT_ARTICLE_CACHE, dict) and len(_GPT_ARTICLE_CACHE) > 20000:
            items = sorted(_GPT_ARTICLE_CACHE.items(), key=lambda kv: int(kv[1].get('ts', 0)), reverse=True)
            trimmed = dict(items[:20000])
        else:
            trimmed = _GPT_ARTICLE_CACHE
        with open(_GPT_ARTICLE_CACHE_PATH, 'w') as _f:
            json.dump(trimmed, _f)
    except Exception:
        pass


def _fetch_article_text(url: str) -> str:
    """Attempt to fetch article HTML and extract main text using BeautifulSoup.

    This is a best-effort heuristic: prefer <article> or <main>, otherwise concat <p> tags.
    Returns empty string on failure.
    """
    if not url:
        return ''
    try:
        r = _http_get(url, timeout=6)
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(r.text, 'html.parser')
        # prefer article tag
        article = soup.find('article')
        if article:
            ps = article.find_all('p')
            text = '\n\n'.join([p.get_text(separator=' ', strip=True) for p in ps if p.get_text(strip=True)])
            if text:
                return text
        main = soup.find('main')
        if main:
            ps = main.find_all('p')
            text = '\n\n'.join([p.get_text(separator=' ', strip=True) for p in ps if p.get_text(strip=True)])
            if text:
                return text
        # fallback: collect longest continuous block of <p>
        ps = soup.find_all('p')
        if ps:
            text = '\n\n'.join([p.get_text(separator=' ', strip=True) for p in ps if p.get_text(strip=True)])
            return text
        # last resort: meta description
        meta = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta and meta.get('content'):
            return meta.get('content')
    except Exception:
        return ''
    return ''


def _openai_sentiment_batch(texts: list) -> list:
    """Compute sentiment scores for multiple texts in a single OpenAI call when possible.

    Returns list of floats in same order as texts. Uses cache when available.
    """
    global _OPENAI_BATCH_CALLS
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return [_simple_sentiment(t) for t in texts]
    results = []
    to_call = []
    idx_map = []
    arr = []
    for i, t in enumerate(texts):
        k = _text_key(t[:1000])
        entry = _SENTIMENT_CACHE.get(k)
        if entry is not None and (int(datetime.now(timezone.utc).timestamp()) - int(entry.get('ts', 0)) < _SENTIMENT_CACHE_TTL):
            results.append(float(entry.get('score', 0.0)))
        else:
            results.append(None)
            to_call.append(t)
            idx_map.append((i, k))

    if to_call:
        combined_prompt = (
            "For each of the following texts, return a JSON array of numeric sentiment scores between -1 and 1 in the same order.\n\n"
            + "\n---\n".join([f"Text {i+1}: " + tt[:1000] for i, tt in enumerate(to_call)])
        )
        try:
            txt = call_chat_completion(
                'You provide numeric sentiment scores for multiple passages.',
                combined_prompt,
                max_output_tokens=384,
                temperature=0.0,
            )
            _OPENAI_BATCH_CALLS += 1
            import re

            m = re.search(r"\[.*\]", txt, re.S)
            if m:
                arr = json.loads(m.group(0))
            else:
                nums = re.findall(r"-?\d+\.?\d*", txt)
                arr = [float(n) for n in nums]
        except Exception:
            arr = [_simple_sentiment(t) for t in to_call]
    # write into results and cache (with timestamps)
    for (i, k), val in zip(idx_map, arr):
        v = max(-1.0, min(1.0, float(val)))
        results[i] = v
        _SENTIMENT_CACHE[k] = {
            'score': v,
            'ts': int(datetime.now(timezone.utc).timestamp()),
            'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        }
    if to_call:
        _save_sentiment_cache()
    # final fallback
    return [r if r is not None else 0.0 for r in results]


def _simple_sentiment(text: str) -> float:
    """Very small keyword-based sentiment heuristic: returns -1..1."""
    if not text:
        return 0.0
    text_l = text.lower()
    score = 0
    positives = ['upgrade', 'beat', 'outperform', 'buy', 'raise', 'strong', 'positive', 'outlook']
    negatives = ['downgrade', 'miss', 'sell', 'lower', 'weak', 'negative', 'cut', 'warn']
    for p in positives:
        if p in text_l:
            score += 1
    for n in negatives:
        if n in text_l:
            score -= 1
    # normalize
    if score > 0:
        return min(1.0, score / 3.0)
    if score < 0:
        return max(-1.0, score / 3.0)
    return 0.0


_CATALYST_KEYWORDS = {
    'earnings/guidance': ['earnings', 'guidance', 'forecast', 'results', 'revenue', 'profit', 'loss'],
    'upgrades/downgrades': ['upgrade', 'downgrade', 'price target', 'initiated', 'reiterated', 'maintains'],
    'product': ['launch', 'product', 'service', 'device', 'chip', 'platform', 'release'],
    'legal/regulatory': ['lawsuit', 'regulator', 'regulatory', 'ftc', 'sec', 'antitrust', 'fine', 'settlement'],
    'executive': ['ceo', 'cfo', 'chief executive', 'executive', 'resigns', 'steps down', 'appoints', 'hires'],
    'outage': ['outage', 'disruption', 'downtime', 'incident', 'breach', 'cyberattack', 'security incident'],
}


def _detect_catalysts(title: str | None, text: str | None, summary: str | None = None) -> List[str]:
    blob_parts = [title or '', summary or '', text or '']
    blob = ' '.join(blob_parts).lower()
    tags = []
    for tag, keywords in _CATALYST_KEYWORDS.items():
        if any(keyword in blob for keyword in keywords):
            tags.append(tag)
    if not tags:
        tags.append('general')
    return tags[:3]


def fetch_alpaca_intraday_bars(symbol: str, start: datetime | None = None, end: datetime | None = None, timeframe: str = '1Min') -> Optional[pd.DataFrame]:
    key = os.getenv('ALPACA_KEY')
    secret = os.getenv('ALPACA_SECRET')
    if not key or not secret:
        return None
    if end is None:
        end = datetime.now(timezone.utc)
    if start is None:
        et = end.astimezone(ZoneInfo('America/New_York'))
        session_start_et = datetime.combine(et.date(), _time(9, 30), tzinfo=ZoneInfo('America/New_York'))
        start = session_start_et.astimezone(timezone.utc)

    url = f'https://data.alpaca.markets/v2/stocks/{symbol}/bars'
    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}
    params = {
        'timeframe': timeframe,
        'start': start.isoformat().replace('+00:00', 'Z'),
        'end': end.isoformat().replace('+00:00', 'Z'),
        'adjustment': 'raw',
        'limit': 1000,
    }
    bars: list[dict] = []
    next_token = None
    try:
        while True:
            if next_token:
                params['page_token'] = next_token
            resp = _http_get(url, params=params, headers=headers, timeout=10)
            data = resp.json()
            chunk = data.get('bars') or []
            if not chunk:
                break
            bars.extend(chunk)
            next_token = data.get('next_page_token')
            if not next_token:
                break
    except Exception:
        return None

    if not bars:
        return None
    df = pd.DataFrame(bars)
    rename_map = {'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'vw': 'vwap'}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def fetch_polygon_analyst_ratings(symbol: str) -> Dict:
    """Attempt to fetch analyst/ratings information from Polygon.

    The Polygon API has several historical/legacy endpoints; this function is defensive
    and tolerant of multiple response shapes. It returns a dict similar to
    the yfinance-based `fetch_analyst_sentiment` result: `{'counts': {...}, 'latest': 'buy'}`
    If no data is available or POLYGON_KEY is not set, returns an empty dict.
    """
    # Prefer explicit REST API key if provided; fall back to legacy POLYGON_KEY (S3)
    key = os.getenv('POLYGON_API_KEY') or os.getenv('POLYGON_KEY')
    if os.getenv('POLYGON_KEY') and os.getenv('POLYGON_API_KEY') and debug_enabled():
        print('[DEBUG] Both POLYGON_API_KEY and POLYGON_KEY are set; prefer POLYGON_API_KEY for REST calls')
    if not key:
        return {}
    # simple on-disk cache for polygon analyst ratings to avoid repeated API calls
    POLY_CACHE_PATH = POLY_CACHE
    try:
        with open(POLY_CACHE_PATH, 'r') as _f:
            _POLY_CACHE = json.load(_f)
    except Exception:
        _POLY_CACHE = {}
    cache_key = f"analyst::{symbol}"
    # cache TTL in seconds (default 12 hours)
    poly_ttl = int(os.getenv('POLYGON_ANALYST_CACHE_TTL', 60 * 60 * 12))
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if cache_key in _POLY_CACHE:
        entry = _POLY_CACHE.get(cache_key, {})
        if now_ts - int(entry.get('ts', 0)) < poly_ttl:
            return entry.get('value', {})
    # Candidate endpoints to try (defensive):
    endpoints = [
        ('https://api.polygon.io/v2/reference/analysts', True),
        (f'https://api.polygon.io/v1/meta/symbols/{symbol}/analysts', False),
        ('https://api.polygon.io/v3/reference/ratings', True),
    ]
    for base, use_params in endpoints:
        try:
            if use_params:
                params = {'ticker': symbol, 'limit': 50, 'apiKey': key}
                r = _http_get(base, params=params, timeout=10)
            else:
                # embed symbol in path
                params = {'apiKey': key}
                r = _http_get(base, params=params, timeout=10)
            j = r.json()
            # possible shapes: {'results': [...]}, {'ratings': [...]}, or a top-level list
            items = j.get('results') or j.get('ratings') or j.get('data') or j
            if isinstance(items, dict):
                # some responses wrap lists in another key
                items = items.get('items') or items.get('results') or []
            if not items:
                continue
            # normalize items to list
            if not isinstance(items, list):
                # try to coerce single-item dict
                if isinstance(items, dict):
                    items = [items]
                else:
                    continue
            counts = {'buy': 0, 'hold': 0, 'sell': 0}
            latest = None
            entries = []
            # iterate newest-first if possible
            for it in items[:50]:
                if not isinstance(it, dict):
                    continue
                # look for rating text in several possible keys
                rtext = None
                for k in ('rating', 'analyst_rating', 'recommendation', 'action'):
                    if it.get(k):
                        rtext = str(it.get(k))
                        break
                # some providers use 'firm_call' or 'call'
                if not rtext:
                    rtext = str(it.get('call') or it.get('firm_call') or it.get('analyst_call') or '')
                v = (rtext or '').lower()
                if 'buy' in v:
                    counts['buy'] += 1
                    if latest is None:
                        latest = 'buy'
                elif 'hold' in v or 'neutral' in v:
                    counts['hold'] += 1
                    if latest is None:
                        latest = 'hold'
                elif 'sell' in v:
                    counts['sell'] += 1
                    if latest is None:
                        latest = 'sell'

                # extract additional helpful fields when present
                # normalize date into ISO-8601 when possible
                raw_date = it.get('date') or it.get('published_at') or it.get('created_at') or it.get('updated_at') or None
                norm_date = None
                if raw_date is not None:
                    try:
                        # numeric epoch
                        if isinstance(raw_date, (int, float)) or (isinstance(raw_date, str) and raw_date.isdigit()):
                            norm_date = datetime.fromtimestamp(int(float(raw_date)), timezone.utc).isoformat()
                        else:
                            # try common ISO / RFC formats
                            from dateutil import parser as _dparser

                            norm_date = _dparser.parse(str(raw_date)).astimezone(timezone.utc).isoformat()
                    except Exception:
                        norm_date = None

                entry = {
                    'rating_text': rtext,
                    'firm': it.get('firm') or it.get('source') or it.get('provider'),
                    'author': it.get('author') or it.get('analyst') or None,
                    'note': it.get('note') or it.get('notes') or it.get('summary') or None,
                    'date': norm_date,
                }
                entries.append(entry)

            if counts['buy'] or counts['hold'] or counts['sell']:
                result = {'counts': counts, 'latest': latest, 'entries': entries}
                # save to cache
                try:
                    _POLY_CACHE[cache_key] = {'ts': now_ts, 'value': result}
                    with open(POLY_CACHE_PATH, 'w') as _f:
                        json.dump(_POLY_CACHE, _f)
                except Exception:
                    pass
                return result
        except Exception:
            # try next endpoint
            continue
    return {}


def fetch_premarket_info(symbol: str) -> Dict:
    """Fetch premarket quote and simple liquidity info for a symbol.

    Returns dict with keys: open, high, low, close, volume, dollar_volume, bid, ask, spread, sector
    If Polygon is available, attempt Polygon; otherwise fall back to yfinance info.
    """
    # Prefer explicit REST API key if provided; fall back to legacy POLYGON_KEY (S3)
    key = os.getenv('POLYGON_API_KEY') or os.getenv('POLYGON_KEY')
    if os.getenv('POLYGON_KEY') and os.getenv('POLYGON_API_KEY') and debug_enabled():
        print('[DEBUG] Both POLYGON_API_KEY and POLYGON_KEY are set; prefer POLYGON_API_KEY for REST calls')
    out = {'open': None, 'high': None, 'low': None, 'close': None, 'volume': None, 'dollar_volume': None, 'bid': None, 'ask': None, 'spread': None, 'sector': None}
    try:
        if key:
            # try a defensive polygon snapshot endpoint
            url = f'https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}'
            params = {'apiKey': key}
            r = _http_get(url, params=params, timeout=10)
            j = r.json()
            # defensive extraction
            tick = j.get('ticker') or j.get('status') or j
            last = j.get('last') or j.get('day') or j.get('todays_change') or {}
            out['open'] = tick.get('prev_open') or last.get('o') or None
            out['high'] = last.get('h') or None
            out['low'] = last.get('l') or None
            out['close'] = last.get('c') or tick.get('last') or None
            out['volume'] = int(last.get('v') or 0)
            out['bid'] = tick.get('b') or None
            out['ask'] = tick.get('a') or None
            if out['bid'] is not None and out['ask'] is not None:
                try:
                    out['spread'] = abs(float(out['ask']) - float(out['bid'])) / max(0.000001, float(out['close'] or out['ask']))
                except Exception:
                    out['spread'] = None
            # estimate dollar volume
            try:
                out['dollar_volume'] = float(out['close'] or 0.0) * float(out['volume'] or 0)
            except Exception:
                out['dollar_volume'] = None
            # sector may appear in fundamentals
            out['sector'] = j.get('fundamentals', {}).get('sector') or None
            return out
    except Exception:
        pass
    # fallback to yfinance
    try:
        t = yf.Ticker(symbol)
        info = t.info if hasattr(t, 'info') else {}
        out['close'] = info.get('previousClose') or info.get('regularMarketPreviousClose') or out['close']
        out['open'] = info.get('open') or out['open']
        out['sector'] = info.get('sector')
        # best-effort spread using bid/ask
        out['bid'] = info.get('bid')
        out['ask'] = info.get('ask')
        try:
            if out['bid'] is not None and out['ask'] is not None and out['close'] is not None:
                out['spread'] = abs(float(out['ask']) - float(out['bid'])) / max(0.000001, float(out['close']))
        except Exception:
            out['spread'] = None
        # volume not available reliably here; leave None
    except Exception:
        pass
    return out


def _dedupe_and_score_articles(articles: list) -> list:
    """Deduplicate articles by title+url and apply freshness boosting and penalties.

    Adds fields to each article: 'freshness_bonus' (float), 'penalty' (bool) and returns filtered list.
    Freshness: boost articles published within last 4 hours by +0.2 to provider_score.
    Penalties: mark articles containing dilution/offering/guidance-cut keywords as penalty=True.
    """
    seen = set()
    out = []
    now_ts = int(datetime.now(timezone.utc).timestamp())
    for a in articles:
        title = (a.get('title') or '').strip()
        url = a.get('url') or a.get('article_url') or ''
        domain = ''
        if url:
            try:
                domain = urlparse(url).netloc.lower()
            except Exception:
                domain = ''
        a['domain'] = domain
        key = hashlib.sha256((title + (domain or '') + (url or '')).encode('utf-8')).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        pa = a.get('published_at')
        freshness_bonus = 0.0
        try:
            if pa:
                if isinstance(pa, (int, float)) or (isinstance(pa, str) and pa.isdigit()):
                    ts = int(float(pa))
                else:
                    from dateutil import parser as _dparser

                    ts = int(_dparser.parse(str(pa)).astimezone(timezone.utc).timestamp())
                age_hours = (now_ts - ts) / 3600.0
                if age_hours <= 4:
                    freshness_bonus = 0.2
        except Exception:
            freshness_bonus = 0.0

        text = (title + ' ' + (a.get('abstract') or '')).lower()
        penalty = False
        if any(k in text for k in ('dilution', 'offering', 'secondary offering', 'rights offering', 'follow-on')):
            penalty = True
        if any(k in text for k in ('cut guidance', 'lowered guidance', 'guidance cut', 'reduced outlook', 'warn on guidance')):
            penalty = True

        prov = a.get('provider_score')
        if prov is None:
            prov = _simple_sentiment(title)
        try:
            prov = float(prov) + freshness_bonus
            if penalty:
                prov -= 0.5
        except Exception:
            prov = float(_simple_sentiment(title))
        prov = max(-1.0, min(1.0, prov))
        a['provider_score'] = prov
        a['freshness_bonus'] = freshness_bonus
        a['penalty'] = penalty

        # simple source trust weighting (defaults)
        trust = 0.6
        a['source_trust'] = trust

        # compute a basic confidence: combination of trust and freshness (0..1)
        conf = max(0.0, min(1.0, 0.6 * trust + 0.4 * min(1.0, freshness_bonus / 0.2)))
        a['confidence'] = conf

        out.append(a)
    return out


def aggregate_news_for_symbol(symbol: str, limit: int = 20, hours: int | None = None, seed_articles: list | None = None):
    """Aggregate news for a symbol and return normalized list and aggregate sentiment.

    `seed_articles` can be provided to supply article dicts directly (useful for tests).
    Only returns articles published within the last `hours` hours (default 24).
    """
    if hours is None:
        try:
            hours = float(os.getenv('NEWS_WINDOW_HOURS', '36'))
        except Exception:
            hours = 36.0
    half_life_hours = float(os.getenv('NEWS_RECENCY_HALF_LIFE_HOURS', '12')) or 12.0
    confidence_floor = float(os.getenv('NEWS_CONFIDENCE_FLOOR', '0.25'))
    articles = list(seed_articles or [])

    # dedupe articles and apply freshness and penalty adjustments
    articles = _dedupe_and_score_articles(articles)

    # Filter to last N hours
    if hours is not None:
        now_ts = int(datetime.now(timezone.utc).timestamp())
        filtered = []
        for a in articles:
            pa = a.get('published_at')
            ts = None
            try:
                if pa is None:
                    ts = None
                elif isinstance(pa, (int, float)) or (isinstance(pa, str) and pa.isdigit()):
                    ts = int(float(pa))
                else:
                    from dateutil import parser as _dparser

                    ts = int(_dparser.parse(str(pa)).astimezone(timezone.utc).timestamp())
            except Exception:
                ts = None
            if ts is None:
                # if no timestamp, conservatively exclude
                continue
            age_hours = (now_ts - ts) / 3600.0
            if age_hours <= float(hours):
                a['_ts'] = ts
                a['_age_hours'] = age_hours
                filtered.append(a)
        if filtered:
            articles = filtered
        elif articles:
            articles = articles[:limit]

    if articles:
        articles.sort(key=lambda a: a.get('_ts', 0), reverse=True)
        if limit:
            articles = articles[:limit]

    # normalize dates and compute model_score average
    weighted_total = 0.0
    weight_sum = 0.0
    fallback_total = 0.0
    fallback_count = 0
    texts = [a.get('title', '') for a in articles]
    use_batch = os.getenv('OPENAI_USE_BATCHED_SENTIMENT', 'true').lower() in ('1', 'true', 'yes')
    catalyst_counter: Counter[str] = Counter()
    # If OpenAI key is present, call ChatGPT to analyze each article for a 0..100 score
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
    model_scores = []
    if OPENAI_KEY:
        try:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            for a in articles:
                # determine a cache key: use URL when present, otherwise title hash
                url = a.get('url')
                if url:
                    cache_key = hashlib.sha256(url.encode('utf-8')).hexdigest()
                else:
                    cache_key = _text_key((a.get('title') or '')[:1000])

                # check cache
                cached = _GPT_ARTICLE_CACHE.get(cache_key)
                if cached and isinstance(cached, dict) and 'sentiment' in cached:
                    cached_model = cached.get('model')
                    if cached_model and cached_model != model_name:
                        cached = None
                    elif cached_model is None and _is_gpt5_model(model_name):
                        cached = None
                if cached and isinstance(cached, dict) and 'sentiment' in cached:
                    try:
                        if now_ts - int(cached.get('ts', 0)) < _GPT_ARTICLE_CACHE_TTL:
                            sentiment = float(cached.get('sentiment', 0.0))
                            confidence = float(cached.get('confidence', 0.0))
                            catalysts = cached.get('catalysts') if isinstance(cached.get('catalysts'), list) else []
                            score100 = int(cached.get('score', round((sentiment + 1.0) * 50.0)))
                            sentiment = max(-1.0, min(1.0, sentiment))
                            confidence = max(0.0, min(1.0, confidence))
                            score100 = max(0, min(100, score100))
                            a['gpt_sentiment'] = sentiment
                            a['gpt_confidence'] = confidence
                            a['gpt_catalysts'] = catalysts
                            a['gpt_score_100'] = score100
                            a['gpt_sent'] = sentiment
                            tags = catalysts or _detect_catalysts(a.get('title'), a.get('content'), a.get('summary'))
                            a['catalyst_tags'] = tags
                            catalyst_counter.update(tags)
                            model_scores.append(sentiment)
                            continue
                    except Exception:
                        pass

                # obtain article text if not present
                art_text = a.get('article_text') or ''
                if not art_text and url:
                    art_text = _fetch_article_text(url)
                    if art_text:
                        a['article_text'] = art_text

                content = (art_text or (a.get('title', '') + '\n\n' + (a.get('description') or '')))[:5000]
                try:
                    default_temp = float(os.getenv('OPENAI_TEMPERATURE', '0'))
                except Exception:
                    default_temp = 0.0
                temperature = default_temp
                if model_name and model_name.lower().startswith('gpt-5') and temperature == 0.0:
                    temperature = 1.0
                prompt = (
                    "You are an equity news analyst. Assess the article below for its expected 1-5 day price impact on the referenced ticker. "
                    "Return ONLY valid JSON with keys: sentiment (float in [-1,1]), confidence (float in [0,1]), catalysts (array of up to 3 short strings). "
                    "Sentiment should be positive for bullish impact, negative for bearish. Confidence reflects strength of the signal. "
                    "Catalysts should summarize the main drivers or be an empty array.\n\nArticle:\n" + content
                )
                try:
                    txt = call_chat_completion(
                        'You are an equity news analyst providing short JSON sentiment summaries.',
                        prompt,
                        max_output_tokens=320,
                        temperature=temperature,
                    )
                    try:
                        j = json.loads(txt)
                    except Exception:
                        import re

                        m = re.search(r"\{.*\}", txt, re.S)
                        j = json.loads(m.group(0)) if m else {}
                    sentiment = float(j.get('sentiment', 0.0))
                    confidence = float(j.get('confidence', 0.0))
                    catalysts = j.get('catalysts') or []
                    if not isinstance(catalysts, list):
                        catalysts = []
                    catalysts = [str(c)[:120] for c in catalysts[:3]]
                except Exception:
                    sentiment = _simple_sentiment(content)
                    confidence = 0.3
                    catalysts = []

                sentiment = max(-1.0, min(1.0, float(sentiment)))
                confidence = max(0.0, min(1.0, float(confidence)))
                score100 = int(round((sentiment + 1.0) * 50.0))

                # attach and cache
                a['gpt_sentiment'] = sentiment
                a['gpt_confidence'] = confidence
                a['gpt_catalysts'] = catalysts
                a['gpt_score_100'] = max(0, min(100, score100))
                a['gpt_sent'] = sentiment
                tags = catalysts or _detect_catalysts(a.get('title'), a.get('content'), a.get('summary'))
                a['catalyst_tags'] = tags
                catalyst_counter.update(tags)
                model_scores.append(sentiment)
                try:
                    _GPT_ARTICLE_CACHE[cache_key] = {
                        'score': a['gpt_score_100'],
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'catalysts': catalysts,
                        'ts': now_ts,
                        'model': model_name,
                    }
                    _save_gpt_article_cache()
                except Exception:
                    pass
        except Exception:
            # fallback to previous method
            if use_batch:
                model_scores = _openai_sentiment_batch(texts)
            else:
                model_scores = [_openai_sentiment(t) for t in texts]
    else:
        if use_batch:
            model_scores = _openai_sentiment_batch(texts)
        else:
            model_scores = [_openai_sentiment(t) for t in texts]

    for a, ms in zip(articles, model_scores):
        # published_at may be epoch or ISO; try to normalize
        pa = a.get('published_at')
        if isinstance(pa, (int, float)):
            try:
                a['published_at'] = datetime.fromtimestamp(float(pa), timezone.utc).isoformat()
            except Exception:
                a['published_at'] = None
        # provider_score fallback
        if a.get('provider_score') is None:
            a['provider_score'] = _simple_sentiment(a.get('title', ''))

        if 'gpt_sentiment' not in a:
            sentiment = max(-1.0, min(1.0, float(ms)))
            a['gpt_sentiment'] = sentiment
            a['gpt_score_100'] = max(0, min(100, int(round((sentiment + 1.0) * 50.0))))
            a['gpt_sent'] = sentiment
        if not a.get('catalyst_tags'):
            tags = _detect_catalysts(a.get('title'), a.get('article_text') or a.get('content'), a.get('summary'))
            a['catalyst_tags'] = tags
            catalyst_counter.update(tags)
        a['_model_score'] = ms

    aggregate_catalysts = [tag for tag, _ in catalyst_counter.most_common(5)]
    for a in articles:
        a['aggregate_catalysts'] = aggregate_catalysts
        if 'gpt_confidence' not in a or a['gpt_confidence'] is None:
            a['gpt_confidence'] = max(confidence_floor, 0.4)
        if 'gpt_catalysts' not in a:
            a['gpt_catalysts'] = []

        a['model_score'] = a.get('_model_score')
        # combine provider and model if both exist
        combined = 0.0
        weight_components = 0.0
        if a.get('provider_score') is not None:
            combined += 0.3 * float(a['provider_score'])
            weight_components += 0.3
        if a.get('model_score') is not None:
            combined += 0.7 * float(a['model_score'])
            weight_components += 0.7
        if weight_components > 0:
            combined = combined / weight_components
        else:
            combined = 0.0
        a['combined_score'] = combined

        age_hours = a.get('_age_hours')
        if age_hours is None:
            weight = 0.0
        else:
            decay = math.exp(-age_hours / max(half_life_hours, 1e-6)) if half_life_hours > 0 else 1.0
            confidence_weight = max(confidence_floor, min(1.0, float(a.get('gpt_confidence', 0.5))))
            weight = decay * confidence_weight
        a['recency_weight'] = weight
        if weight > 0:
            weighted_total += combined * weight
            weight_sum += weight
        else:
            fallback_total += combined
            fallback_count += 1

    if weight_sum > 0:
        aggregate = weighted_total / weight_sum
    elif fallback_count > 0:
        aggregate = fallback_total / fallback_count
    else:
        aggregate = 0.0

    return articles, aggregate


def _load_cached_eod(symbol: str, output_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(output_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def fetch_eod(symbols: List[str], output_dir: str = './data') -> Dict[str, Optional[pd.DataFrame]]:
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Optional[pd.DataFrame]] = {}
    cache_only = env_bool('EOD_CACHE_ONLY', False)
    for s in symbols:
        cached_df = _load_cached_eod(s, output_dir)
        if cache_only and cached_df is not None:
            results[s] = cached_df
            continue
        t = yf.Ticker(s)
        df = None
        download_error = None
        if not cache_only:
            try:
                df = t.history(period='5d', auto_adjust=False, actions=False, threads=False)
            except Exception as exc:
                download_error = exc
        if df is None or df.empty:
            if cached_df is not None:
                if download_error and debug_enabled():
                    logger.warning('fetch_eod: using cached data for %s after download failure: %s', s, download_error)
                results[s] = cached_df
                continue
            if download_error and debug_enabled():
                logger.warning('fetch_eod: failed to download %s: %s', s, download_error)
            results[s] = None
            continue
        path = os.path.join(output_dir, f"{s}.csv")
        try:
            df.to_csv(path)
        except Exception as exc:
            if debug_enabled():
                logger.warning('fetch_eod: failed to persist CSV for %s: %s', s, exc)
        results[s] = df
    return results


def fetch_analyst_sentiment(symbol: str) -> Dict:
    """Fetch recent analyst recommendations and return a simple sentiment summary.

    Returns a dict like {'buy': 3, 'hold': 1, 'sell': 0, 'latest': 'Buy'}
    If unavailable, returns an empty dict.
    """
    # Prefer Polygon analyst ratings if available (more real-time / unified source)
    try:
        poly = fetch_polygon_analyst_ratings(symbol)
        if poly:
            return poly
    except Exception:
        # fall back to yfinance below
        pass

    try:
        t = yf.Ticker(symbol)
        rec = t.recommendations
        if rec is None or rec.empty:
            return {}
        # recommendations often include a column with recommendation text; consolidate
        text_cols = [c for c in rec.columns if rec[c].dtype == object]
        # flatten recent 20 rows
        recent = rec.tail(20).astype(str).apply(lambda row: ' '.join(row.values), axis=1)
        counts = {'buy': 0, 'hold': 0, 'sell': 0}
        latest = None
        for i, val in enumerate(recent[::-1]):
            v = val.lower()
            if 'buy' in v:
                counts['buy'] += 1
                if latest is None:
                    latest = 'buy'
            elif 'hold' in v or 'neutral' in v:
                counts['hold'] += 1
                if latest is None:
                    latest = 'hold'
            elif 'sell' in v:
                counts['sell'] += 1
                if latest is None:
                    latest = 'sell'
        return {'counts': counts, 'latest': latest}
    except Exception:
        return {}
# module logger
logger = logging.getLogger(__name__)
