# Trading Bot (minimal)

This repository contains a tiny Alpaca trading example script `main.py`.

Setup

1. Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Environment

Create a `.env` file in the project root or export these variables in your shell:

```
ALPACA_KEY=your_key_here
ALPACA_SECRET=your_secret_here
ALPACA_PAPER=true    # use paper trading endpoint
DRY_RUN=true         # default true; set to false to actually send an order
TEST_SYMBOL=AAPL
TEST_QTY=1
TEST_SIDE=buy
```

Run

```bash
python3 main.py
```

Automation
----------

This repository includes `scripts/run_scheduled_trader.sh`, which wraps `scheduled_trader.py --invest`, forces the non-interactive confirmation (`INVEST_FORCE=1`), and writes stdout/stderr to `logs/scheduled_trader.log`. The script automatically prefers `./.venv/bin/python` or `./venv/bin/python` when present, falling back to `python3` (or the value provided in `PYTHON_BIN`). `scripts/scheduled_trader_cron.sh` builds on that wrapper for cron usage: it copies the latest `data/mag7_analysis.json` and `data/investment_action.json` into timestamped `logs/data_snapshots/` files and appends a line to `logs/cron.log` after each run. You can run the automation unattended either on your Mac or by pushing the project to GitHub.

### macOS launchd

1. Copy `docs/launchd/com.tradingbot.scheduledtrader.plist.example` to `~/Library/LaunchAgents/com.tradingbot.scheduledtrader.plist`.
2. Edit the copied plist so all `/Users/your_user/Desktop/trading-bot/...` paths match your actual project location.
3. Adjust the `Hour`/`Minute` under `StartCalendarInterval` for the time (local time) you want the automation to start. The example is 08:45 Eastern.
4. Load the agent with `launchctl load ~/Library/LaunchAgents/com.tradingbot.scheduledtrader.plist`. Use `launchctl unload` to stop or `launchctl list | grep tradingbot` to confirm.
5. Check `logs/scheduled_trader.log` and the `StandardOutPath`/`StandardErrorPath` defined in the plist to monitor execution. To test immediately, run `launchctl kickstart -k gui/$UID/com.tradingbot.scheduledtrader` after loading.

### GitHub Actions

1. Commit the repository to GitHub; the workflow file `.github/workflows/scheduled-trader.yml` will trigger a job named **Scheduled Trader**.
2. In the repository Settings → Secrets and variables → Actions screen, add the secrets the workflow expects: `ALPACA_KEY`, `ALPACA_SECRET`, `OPENAI_API_KEY` (plus optional overrides like `OPENAI_MODEL`, `OPENAI_WEB_MODEL`, `PLACE_ORDER`, `USE_WEB_RESEARCH`, `REALLOCATE_FULL`, `TOP_ALLOCATE_COUNT`, or `SCHEDULED_TRADER_MODE`).
3. Adjust the `cron` expression under `schedule` in `.github/workflows/scheduled-trader.yml` if you need a different start time or frequency. The default schedules 09:00 ET (pre-open) and 16:30 ET (post-close) on weekdays, with an ET-time guard enabled to handle DST.
4. The workflow installs dependencies, runs `scripts/run_scheduled_trader.sh`, and uploads `logs/scheduled_trader.log` plus JSON artifacts under `artifacts/` as run artifacts. Inspect the artifact or job log after each run to confirm behavior.
5. You can run the job on demand from the Actions tab using **Run workflow** (it honors the same secrets and environment variables).

### Cron (macOS/Linux)

1. Ensure the script is executable (`chmod +x scripts/scheduled_trader_cron.sh`) and that your `.env` contains the required API keys.
2. Determine the absolute path to the project root, e.g. `/Users/your_user/Desktop/trading-bot`.
3. Edit your crontab with `crontab -e` and add:

   `*/15 * * * * /bin/bash -lc '/Users/your_user/Desktop/trading-bot/scripts/scheduled_trader_cron.sh'`

   Replace the path with your own. The `-lc` ensures your login shell runs so that `.env` and PATH exports inside the script resolve correctly.
4. Cron will invoke the script every 15 minutes. Review `logs/scheduled_trader.log` for run output, `logs/cron.log` for a one-line status per interval, and `logs/data_snapshots/` for timestamped copies of the data files.
5. To disable, remove the line with `crontab -e`. If you need to run manually, execute `bash scripts/scheduled_trader_cron.sh` from the project root.

Notes

- The script defaults to dry-run to avoid accidental live orders.
- Use `DRY_RUN=false` and `ALPACA_PAPER=false` only when you're sure.
- Set `EOD_CACHE_ONLY=1` to reuse previously downloaded CSVs and skip live price fetches (handy for offline cron runs or when Yahoo Finance is unreachable).

Placing orders

- To actually place an order you must explicitly opt-in. You can either set the environment variable:

```
PLACE_ORDER=true
```

or run with the CLI flag `--place`. For safety the script will prompt for confirmation; use `--yes` to skip the prompt.

Example (paper trading):

```bash
ALPACA_KEY=... ALPACA_SECRET=... PLACE_ORDER=true python3 main.py
```

or

```bash
ALPACA_KEY=... ALPACA_SECRET=... python3 main.py --place --yes
```

For local testing without valid Alpaca keys you can skip the account check with `SKIP_ACCOUNT_CHECK=true`.

Polygon integration
-------------------

This project can optionally pull analyst ratings and premarket snapshots from Polygon.io. To enable Polygon support set the `POLYGON_KEY` environment variable in your `.env` or shell (POLYGON_KEY is the canonical name). When present the bot will:

- Prefer `fetch_polygon_analyst_ratings` when computing analyst sentiment; this tries a few Polygon endpoints and normalizes ratings into `{'counts': {...}, 'latest': 'buy'|'hold'|'sell', 'entries': [...]}`.
- Use Polygon snapshots in `fetch_premarket_info` when available (fallback is yfinance).

Example env vars:

```
POLYGON_KEY=your_polygon_api_key_here
```

OpenAI model selection
----------------------

You can select which OpenAI model the bot uses by setting `OPENAI_MODEL` in your `.env`. The default in `.env.example` is `gpt-5.2`. Be careful: using larger models may increase latency and cost.

If Polygon is not available the code will fall back to yfinance for analyst recommendations.

OpenAI / ChatGPT integration (news/article scoring)
--------------------------------------------------

This project can optionally use OpenAI (ChatGPT) to analyze news articles and provide per-article sentiment and higher-level portfolio analysis. When `USE_WEB_RESEARCH=1`, it uses OpenAI web search per symbol and does not fetch external news feeds.

Environment variables:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini   # model used for scoring; defaults to gpt-4o-mini
OPENAI_USE_BATCHED_SENTIMENT=true  # when true uses the batched -1..1 sentiment endpoint for fallbacks
GPT_ARTICLE_CACHE_TTL=1209600  # (seconds) TTL for per-article GPT cache; default 14 days (1209600s)
```

Behavior and defaults:

- The article aggregator (`aggregate_news_for_symbol` in `eod_fetcher.py`) defaults to a 48-hour window for filtering articles provided via `seed_articles`.
- For each article found, if `OPENAI_API_KEY` is set the system will attempt to fetch the article HTML (when a URL is available), extract the main text using BeautifulSoup, and call the OpenAI chat completions API to request a numeric score 0..100 for the article.
- Per-article fields added to article dicts:
	- `gpt_score_100` (int): 0..100 score produced by the model
	- `gpt_sent` (float): normalized sentiment in [-1.0, 1.0] computed as `(gpt_score_100 - 50)/50`
	- `article_text` (str): extracted HTML text when available

Caching and cost-savings:

- The system caches per-article GPT results in `.cache/gpt_article_cache.json` keyed by article URL (SHA256) or title-hash when URL missing. The cache TTL is controlled by `GPT_ARTICLE_CACHE_TTL` (default 14 days). This reduces repeated API calls and cost.
- The OpenAI client is instantiated lazily on cache-miss to avoid unnecessary client creation.

Dashboard (live portfolio + manual reallocation)
------------------------------------------------

This repo includes a small dashboard server that shows the Mag7 watchlist, current portfolio, and the latest ranking.

Run locally:

```bash
python3 dashboard_server.py
```

Optional env vars:

```
DASHBOARD_TOKEN=your_token_here   # if set, required for manual reallocation
DASHBOARD_PORT=8000
PLACE_ORDER=true                 # required for manual reallocation to place trades
USE_WEB_RESEARCH=1
REALLOCATE_FULL=1
```

The manual reallocation button calls `/api/reallocate` on the server. If `DASHBOARD_TOKEN` is set, the UI must send it as a Bearer token.

Fallbacks and safety:

- If `OPENAI_API_KEY` is not set, the system falls back to heuristic sentiment (`_simple_sentiment`) and a batched -1..1 OpenAI sentiment method if enabled.
- Fetching arbitrary URLs has risks (network, SSRF); the fetcher uses a 6s timeout and conservative extraction, but consider adding a domain allowlist or other network policy for production.
- The model parsing is defensive: it attempts JSON extraction and numeric fallback; you should review logs for parsing errors if you change prompts or models.

Testing
-------

Unit tests were added to ensure the ChatGPT scoring and caching behavior. The tests mock OpenAI and HTTP fetches so they run without real API keys.

If you'd like, I can: add stricter schema validation for model output, integrate a more advanced article extractor, or add an allowlist for fetched domains.
