# Trading Bot

**Use this folder (`trading-bot`) as the only project directory.** All code, scripts, and config live here—not in "trading-bot 2" or elsewhere.

Automated trading bot with email alerts. Can run locally or on **GitHub Actions** (schedule or manual).

## Run on GitHub

1. **Add secrets** in the repo: **Settings** → **Secrets and variables** → **Actions**.  
   See [GITHUB_SETUP.md](GITHUB_SETUP.md) for the full list (Alpaca, OpenAI, and optional email alert secrets).

2. **Manual run**: **Actions** → **Scheduled Trader** → **Run workflow** → choose mode (`mag7`, `invest`, or `close`) → **Run workflow**.

3. **Schedule**: The workflow runs at 09:00 ET and 16:30 ET on weekdays. Configure `SCHEDULED_TRADER_MODE` in secrets if you want a different default than `mag7`.

4. **Email alerts on GitHub**: Add secrets `ALERT_EMAIL_TO`, `ALERT_SMTP_HOST`, `ALERT_SMTP_PORT`, `ALERT_SMTP_USER`, `ALERT_SMTP_PASSWORD` so the bot can send alerts when it runs in Actions.

**Note:** The workflow needs `eod_fetcher.py`, `indicators.py`, and `analyze_with_chatgpt.py` in the repo. If you see an `ImportError` for those, add those files and push.

## Email alerts (local or GitHub)

The email feature in `alerts.py` works when the right env vars are set (locally in `.env`, or in GitHub as secrets):

- `ALERT_EMAIL_TO`, `ALERT_SMTP_HOST`, `ALERT_SMTP_PORT`, `ALERT_SMTP_USER`, `ALERT_SMTP_PASSWORD`

Behavior: sends an email when there are stock ratings and/or orders (including when the outcome is all hold), with a per-symbol summary and analysis snippet.

## Test email locally

```bash
pip install -r requirements.txt
python scripts/test_email_alert.py
```

Requires a `.env` with the `ALERT_*` variables.

## Repo contents

- `alerts.py` – email alert logic
- `scheduled_trader.py` – main trader (invest / Mag7 / close)
- `project_config.py` – paths
- `scripts/run_scheduled_trader.sh` – wrapper for mag7 / invest / close
- `.github/workflows/scheduled-trader.yml` – GitHub Action (schedule + manual)
- [GITHUB_SETUP.md](GITHUB_SETUP.md) – how to configure secrets and run on GitHub
