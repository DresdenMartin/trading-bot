import json
import os
import threading
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory, abort

import requests
import yfinance as yf

import scheduled_trader
from scheduled_trader import env_bool


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_ROOT, "dashboard")
ARTIFACTS_DIR = os.path.join(APP_ROOT, "artifacts")
LOGS_DIR = os.path.join(APP_ROOT, "logs")
HISTORY_PATH = os.path.join(LOGS_DIR, "portfolio_history.jsonl")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
_REBALANCE_LOCK = threading.Lock()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj):
    if isinstance(obj, requests.Response):
        return scheduled_trader._normalize_order_resp(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _append_portfolio_history(acct: dict | None, positions: list | None):
    os.makedirs(LOGS_DIR, exist_ok=True)
    min_seconds = int(os.getenv("PORTFOLIO_HISTORY_MIN_SECONDS", "300"))
    now_ts = int(datetime.now(timezone.utc).timestamp())
    last_ts = None
    try:
        with open(HISTORY_PATH, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size > 2:
                fh.seek(max(0, size - 2048), os.SEEK_SET)
                tail = fh.read().decode("utf-8", errors="ignore").strip().splitlines()
                if tail:
                    last = json.loads(tail[-1])
                    last_ts = int(last.get("ts", 0))
    except Exception:
        last_ts = None

    if last_ts and now_ts - last_ts < min_seconds:
        return

    acct = acct or {}
    equity = acct.get("equity") or acct.get("portfolio_value")
    cash = acct.get("cash")
    try:
        equity = float(equity) if equity is not None else None
    except Exception:
        equity = None
    try:
        cash = float(cash) if cash is not None else None
    except Exception:
        cash = None

    entry = {
        "ts": now_ts,
        "timestamp": _now_iso(),
        "equity": equity,
        "cash": cash,
        "positions": positions or [],
    }
    try:
        with open(HISTORY_PATH, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _auth_ok():
    token = os.getenv("DASHBOARD_TOKEN")
    if not token:
        return True
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        provided = header.split(" ", 1)[1]
    else:
        provided = request.headers.get("X-API-Key", "")
    return provided == token


def _alpaca_headers():
    key = os.getenv("ALPACA_KEY")
    secret = os.getenv("ALPACA_SECRET")
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _alpaca_base():
    return "https://paper-api.alpaca.markets" if env_bool("ALPACA_PAPER", True) else "https://api.alpaca.markets"


def _alpaca_latest_trade(symbol: str):
    headers = _alpaca_headers()
    if not headers:
        return None
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        trade = data.get("trade") or {}
        price = trade.get("p") or trade.get("price")
        ts = trade.get("t") or trade.get("timestamp")
        return {"price": float(price), "timestamp": ts}
    except Exception:
        return None


def _yf_latest_price(symbol: str):
    try:
        t = yf.Ticker(symbol)
        info = getattr(t, "fast_info", {}) or {}
        price = info.get("lastPrice") or info.get("last_price")
        if price is not None:
            return float(price)
    except Exception:
        pass
    return None


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/history")
def history():
    return send_from_directory(STATIC_DIR, "history.html")


@app.get("/api/portfolio")
def api_portfolio():
    acct, positions = scheduled_trader.fetch_alpaca_account_and_positions()
    _append_portfolio_history(acct, positions)
    return jsonify({"timestamp": _now_iso(), "account": acct, "positions": positions})


@app.get("/api/watchlist")
def api_watchlist():
    symbols = scheduled_trader.get_symbols()
    out = []
    for sym in symbols:
        trade = _alpaca_latest_trade(sym)
        price = trade.get("price") if trade else None
        ts = trade.get("timestamp") if trade else None
        if price is None:
            price = _yf_latest_price(sym)
        out.append({"symbol": sym, "price": price, "timestamp": ts})
    return jsonify({"timestamp": _now_iso(), "symbols": out})


@app.get("/api/analysis")
def api_analysis():
    path = os.path.join(ARTIFACTS_DIR, "mag7_analysis.json")
    if not os.path.exists(path):
        return jsonify({"timestamp": _now_iso(), "analysis": None})
    try:
        data = json.load(open(path, "r"))
    except Exception:
        data = None
    return jsonify({"timestamp": _now_iso(), "analysis": data})


@app.get("/api/open_orders")
def api_open_orders():
    headers = _alpaca_headers()
    if not headers:
        return jsonify({"error": "missing alpaca keys"}), 400
    base = _alpaca_base()
    try:
        resp = requests.get(
            f"{base}/v2/orders",
            headers=headers,
            params={"status": "open", "limit": 200, "nested": "true"},
            timeout=10,
        )
        resp.raise_for_status()
        orders = resp.json()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"timestamp": _now_iso(), "orders": orders})


@app.post("/api/cancel_open_orders")
def api_cancel_open_orders():
    if not _auth_ok():
        abort(401)
    headers = _alpaca_headers()
    if not headers:
        return jsonify({"error": "missing alpaca keys"}), 400
    base = _alpaca_base()
    try:
        resp = requests.delete(f"{base}/v2/orders", headers=headers, timeout=10)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:
            payload = {"status": "ok"}
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"timestamp": _now_iso(), "cancelled": payload})


@app.get("/api/history")
def api_history():
    limit = int(request.args.get("limit", "240"))
    rows = []
    try:
        with open(HISTORY_PATH, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        rows = []
    if limit and len(rows) > limit:
        rows = rows[-limit:]
    return jsonify({"timestamp": _now_iso(), "history": rows})


@app.post("/api/reallocate")
def api_reallocate():
    if not _auth_ok():
        abort(401)
    if not env_bool("PLACE_ORDER", False):
        return jsonify({"error": "PLACE_ORDER is not enabled"}), 400
    if not _REBALANCE_LOCK.acquire(blocking=False):
        return jsonify({"error": "reallocation already running"}), 409
    try:
        result = scheduled_trader.analyze_mag7_and_invest(
            reallocate_full_arg=True,
            place_order_arg=True,
            invest_yes_arg=True,
        )
        payload = json.dumps(result, default=_json_default)
        return app.response_class(payload, mimetype="application/json")
    finally:
        _REBALANCE_LOCK.release()


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
