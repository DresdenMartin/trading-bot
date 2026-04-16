"""
Path constants for cache and data files used across the trading bot modules.
Reconstructed from import references in scheduled_trader.py, eod_fetcher.py,
and analyze_with_chatgpt.py.
"""
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_BASE, '.cache')
DATA_DIR = os.path.join(_BASE, 'data')

# ── Cache files ──────────────────────────────────────────────────────────────
GPT_ARTICLE_CACHE   = os.path.join(CACHE_DIR, 'gpt_article_cache.json')
SENTIMENT_CACHE     = os.path.join(CACHE_DIR, 'sentiment_cache.json')
POLY_CACHE          = os.path.join(CACHE_DIR, 'polygon_analyst_cache.json')
WEB_RESEARCH_CACHE  = os.path.join(CACHE_DIR, 'web_research_cache.json')

# ── Data / audit files ───────────────────────────────────────────────────────
ORDER_AUDIT         = os.path.join(DATA_DIR, 'order_audit.jsonl')
SAFE_ORDERS_LOG     = os.path.join(DATA_DIR, 'safe_orders.log')
MAG7_ANALYSIS       = os.path.join(DATA_DIR, 'mag7_analysis.json')
MAG7_REALLOC_PLAN   = os.path.join(DATA_DIR, 'mag7_reallocation_plan.json')
WEB_RESEARCH_AUDIT  = os.path.join(DATA_DIR, 'web_research_audit.jsonl')
