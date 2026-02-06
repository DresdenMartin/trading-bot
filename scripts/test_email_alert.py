#!/usr/bin/env python3
"""Send a test trade alert email. Run from project root with .env set."""
import os
import sys

# Allow importing alerts from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from alerts import send_trade_alert

if not os.getenv("ALERT_EMAIL_TO"):
    print("Set ALERT_EMAIL_TO (and other ALERT_* vars) in .env")
    sys.exit(1)

# Minimal result that triggers an email (stock ratings)
result = {
    "candidates": [
        {"symbol": "AAPL", "suggested": "hold", "score": 50, "confidence": 60, "latest_price": 150.0},
        {"symbol": "MSFT", "suggested": "buy", "score": 72, "confidence": 80, "latest_price": 380.0},
    ],
    "note": "Test run – email feature check",
}
send_trade_alert("test", result)
print("If no error above, check your inbox for the test alert.")
