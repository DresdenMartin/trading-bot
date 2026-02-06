# Trading Bot Logic Overview

## 1. Data Preparation
- Load environment configuration and prepare cache directories.
- Pull recent market data (price history, premarket metrics) for the fixed "Mag Seven" universe (AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA).
- Gather supporting signals: news feeds, analyst ratings, social sentiment, and heuristic sentiment scores.

## 2. Signal Analysis
- Enrich price data with technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands).
- Aggregate and deduplicate news items; normalize timestamps and sentiment scores.
- Use OpenAI scoring when available; fall back to deterministic heuristics when the model is unavailable.
- Combine technical, news, and analyst inputs into per-symbol metrics for the decision engine.

## 3. Portfolio Scoring
- Use the configured Mag Seven list as the candidate universe.
- Run the portfolio analysis prompt to rank symbols 1–100 by short-term outlook.
- Preserve the top set (default 3) for allocation decisions and record supporting rationale.

## 4. Allocation Planning
- Fetch current account balances and open positions from Alpaca.
- Determine whether full reallocation or incremental buying is requested (ENV flags or CLI args).
- Calculate per-symbol dollar targets using configured investment percentages and portfolio value for the top three symbols.
- Produce a dry-run plan when trading is disabled, exporting the analysis for review.

## 5. Order Execution
- For reallocation: submit market sells for symbols outside the target trio, then buys for the leaders.
- Automatically retry buys with notional orders when quantity-based orders are rejected (e.g., 422).
- For incremental investing: place market buys with idempotent client order IDs and extended-hours enabled when requested.
- Log every attempted order (flow, action, symbol, quantity, status) and append audit records to JSONL.

## 6. Safety & Observability
- Respect confirmation prompts unless explicitly bypassed (INVEST_YES / INVEST_FORCE).
- Capture telemetry counters for OpenAI calls and sentiment cache usage.
- Persist generated reports (Mag-7 analysis, allocation plans, investment actions, close actions) under `./data`/`artifacts` for auditing.
- Emit human-readable logging so operators can trace each trade decision and API response.

## 7. Testing & Fallbacks
- Provide heuristic scoring paths to keep the pipeline operational when external models are unavailable.
- Cover reallocation workflows with automated tests, including idempotency and fractional-order fallbacks.
- Mock external services in tests to avoid live API dependencies and ensure deterministic behavior.
