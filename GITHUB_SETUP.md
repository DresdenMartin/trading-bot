# GitHub Actions Setup – Run the Trading Bot Automatically

This guide walks you through setting up the trading bot to run on GitHub's servers via Actions. No local machine or VPS needed.

## 0. Ensure project_config.py is in the repo

The bot requires `project_config.py` (paths for cache, artifacts, logs). It defines `CACHE_DIR`, `GPT_ARTICLE_CACHE`, `MAG7_ANALYSIS`, etc.

## 1. Push Your Repo to GitHub

If you haven't already:

```bash
git remote add origin https://github.com/YOUR_USERNAME/trading-bot.git
git branch -M main
git push -u origin main
```

(Use your actual GitHub username and repo URL.)

## 2. Add Required Secrets

1. Open your repo on GitHub.
2. Go to **Settings** → **Secrets and variables** → **Actions**.
3. Click **New repository secret** for each of these:

| Secret Name       | Required | Description                                      |
|-------------------|----------|--------------------------------------------------|
| `ALPACA_KEY`      | Yes      | Your Alpaca API key ID                           |
| `ALPACA_SECRET`   | Yes      | Your Alpaca API secret                           |
| `OPENAI_API_KEY`  | Yes      | Your OpenAI API key (for news scoring)           |
| `ALPACA_PAPER`    | No       | Set to `true` for paper trading (recommended)    |
| `PLACE_ORDER`     | No       | Set to `true` to place real orders (default off) |

## 3. Email Alert Secrets (optional)

To get trade alerts by email when the workflow runs, add these secrets:

| Secret Name            | Description                          |
|------------------------|--------------------------------------|
| `ALERT_EMAIL_TO`       | Email address to receive alerts      |
| `ALERT_SMTP_HOST`      | e.g. `smtp.gmail.com`                |
| `ALERT_SMTP_PORT`      | e.g. `587`                           |
| `ALERT_SMTP_USER`      | Your email (SMTP login)              |
| `ALERT_SMTP_PASSWORD`  | App password (for Gmail, use App Password) |

## 4. Optional Secrets

Add these only if you use the corresponding features:

| Secret Name             | Description                                              |
|-------------------------|----------------------------------------------------------|
| `POLYGON_KEY`           | Polygon.io API key for analyst ratings and premarket     |
| `OPENAI_MODEL`          | Model override (e.g. `gpt-4o-mini`)                      |
| `USE_WEB_RESEARCH`      | Set to `1` for OpenAI web search per symbol              |
| `REALLOCATE_FULL`       | Set to `1` for full portfolio reallocation               |
| `TOP_ALLOCATE_COUNT`    | Number of top symbols to allocate (default 3)            |
| `SCHEDULED_TRADER_MODE` | `mag7`, `invest`, or `close` (for scheduled runs)        |

## 5. Schedule and Manual Runs

**Automatic schedule**

The workflow runs:

- **09:00 ET** (pre-market)
- **16:30 ET** (post-market)

on weekdays. Times are adjusted for daylight saving.

**Manual run**

1. Go to the **Actions** tab.
2. Select **Scheduled Trader**.
3. Click **Run workflow**.
4. Choose **Run mode**: `mag7`, `invest`, or `close`.
5. Click **Run workflow** (green button).

## 6. View Results

After each run:

1. Open the **Actions** tab.
2. Click the latest run.
3. Download the **trader-logs-XXX** artifact to see logs, analysis, and order artifacts.

## 7. Safety Notes

- Start with `ALPACA_PAPER=true` and `PLACE_ORDER` unset (or `false`) so no live orders are placed.
- Only add `PLACE_ORDER=true` after you're satisfied with paper trading.
- Artifacts are kept for 7 days.
- Review the logs periodically to confirm behavior.

## Troubleshooting

- **"Missing ALPACA_KEY"**: Add all required secrets under Settings → Secrets and variables → Actions.
- **Workflow fails**: Check the Actions run log and ensure your keys are correct and have the needed permissions.
- **No orders placed**: Ensure `PLACE_ORDER` is set to `true` if you intend to place orders (and you've validated behavior in paper mode).
- **ImportError (eod_fetcher, indicators, analyze_with_chatgpt)**: The bot needs these Python modules in the repo. Add `eod_fetcher.py`, `indicators.py`, and `analyze_with_chatgpt.py` to the project root and push.
