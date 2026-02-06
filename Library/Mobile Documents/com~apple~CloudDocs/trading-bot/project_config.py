"""Project paths and config constants."""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, '.cache')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

GPT_ARTICLE_CACHE = os.path.join(CACHE_DIR, 'gpt_article_cache.json')
SENTIMENT_CACHE = os.path.join(CACHE_DIR, 'sentiment_cache.json')
POLY_CACHE = os.path.join(CACHE_DIR, 'poly_cache.json')
WEB_RESEARCH_CACHE = os.path.join(CACHE_DIR, 'web_research_cache.json')
WEB_RESEARCH_AUDIT = os.path.join(LOGS_DIR, 'web_research_audit.jsonl')

SAFE_ORDERS_LOG = os.path.join(LOGS_DIR, 'safe_orders.log')
ORDER_AUDIT = os.path.join(LOGS_DIR, 'order_audit.jsonl')
MAG7_ANALYSIS = os.path.join(ARTIFACTS_DIR, 'mag7_analysis.json')
MAG7_REALLOC_PLAN = os.path.join(ARTIFACTS_DIR, 'mag7_reallocation_plan.json')
