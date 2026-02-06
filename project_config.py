"""Project paths and config for the trading bot."""
import os

# Data and cache dirs (relative to project root or CWD)
DATA_DIR = os.environ.get("TRADING_BOT_DATA", os.path.join(os.getcwd(), "data"))
CACHE_DIR = os.environ.get("TRADING_BOT_CACHE", os.path.join(DATA_DIR, "cache"))
LOGS_DIR = os.environ.get("TRADING_BOT_LOGS", os.path.join(os.getcwd(), "logs"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Artifacts
GPT_ARTICLE_CACHE = os.path.join(CACHE_DIR, "gpt_article_cache.json")
MAG7_ANALYSIS = os.path.join(DATA_DIR, "mag7_analysis.json")
MAG7_REALLOC_PLAN = os.path.join(DATA_DIR, "mag7_realloc_plan.json")
ORDER_AUDIT = os.path.join(DATA_DIR, "order_audit.jsonl")
SAFE_ORDERS_LOG = os.path.join(DATA_DIR, "safe_orders.log")
