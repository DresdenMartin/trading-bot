# Trading Bot

Automated trading bot with email alerts.

## Email alerts

The **email feature is implemented** in `alerts.py` and will work when:

1. **Environment variables** are set (e.g. in `.env`):
   - `ALERT_EMAIL_TO` – recipient address
   - `ALERT_SMTP_HOST` (e.g. `smtp.gmail.com`)
   - `ALERT_SMTP_PORT` (e.g. `587`)
   - `ALERT_SMTP_USER` – your email
   - `ALERT_SMTP_PASSWORD` – app password (for Gmail, use an [App Password](https://support.google.com/accounts/answer/185833))

2. **Your main trading script** imports and calls:
   ```python
   from alerts import send_trade_alert
   send_trade_alert('invest_flow', result)  # or 'analyze_mag7_and_invest', 'reallocate_full', 'close_all'
   ```

Behavior:

- Sends an email when there are **stock ratings and/or orders** (including when the outcome is all **hold**).
- Each email includes a **per-symbol summary**: rating, score, and a short **analysis summary** (and rationale bullets when available).

## Missing code

This repo currently contains:

- `alerts.py` – full email alert logic (ready to use)
- `project_config.py` – paths for data, cache, logs, and artifacts
- `scripts/git-push-with-token.sh` – push to GitHub using a PAT (no password prompt)

The **main trader** (`scheduled_trader.py`) and its dependencies (e.g. `eod_fetcher`, `indicators`, `analyze_with_chatgpt`) are not in this folder. If you have them elsewhere (e.g. iCloud or another clone):

1. Copy `scheduled_trader.py`, `eod_fetcher.py`, `indicators.py`, `analyze_with_chatgpt.py`, and any other required files into this directory.
2. Ensure `scheduled_trader.py` has:
   - `from alerts import send_trade_alert`
   - Calls to `send_trade_alert(flow, result)` after each run (e.g. invest_flow, analyze_mag7_and_invest, reallocate_full, close_all).

Then the email alerts will work with your full pipeline.

## Test email

To confirm SMTP and env are correct without running the full bot:

```bash
python -c "
from dotenv import load_dotenv
load_dotenv()
from alerts import send_trade_alert
send_trade_alert('test', {'candidates': [{'symbol': 'AAPL', 'suggested': 'hold', 'score': 50}], 'note': 'Test run'})
print('Check your inbox.')
"
```

Requires `python-dotenv` and a `.env` with the `ALERT_*` variables.
