import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from eod_fetcher import fetch_eod, fetch_analyst_sentiment, aggregate_news_for_symbol
from indicators import add_sma, add_ema, add_rsi, add_macd, add_bbands
from analyze_with_chatgpt import analyze_summary
import requests
import json
import math


def compute_candidate_score(analysis: dict, metrics: dict, weights: dict = None) -> float:
    """Compute a consolidated score for a candidate symbol.

    Weights is a dict with keys: technical (0..1), news (0..1), model (0..1), analyst (0..1), buy_boost (float)
    Returns normalized score where higher is better.
    """
    if weights is None:
        weights = {'technical': 0.4, 'news': 0.2, 'model': 0.3, 'analyst': 0.1, 'buy_boost': 0.5}
    # technical: derive from simple EMA/SMA/RSI signals presence
    tech_score = 0.0
    try:
        if isinstance(metrics.get('ema_14'), float) and isinstance(metrics.get('sma_14'), float):
            # price above ema and sma -> positive
            if metrics.get('close', 0) > metrics.get('ema_14', 0):
                tech_score += 0.6
            if metrics.get('close', 0) > metrics.get('sma_14', 0):
                tech_score += 0.4
    except Exception:
        pass
    rsi = metrics.get('rsi_14')
    if isinstance(rsi, float):
        if rsi < 30:
            tech_score += 0.6  # oversold -> positive
        elif rsi > 70:
            tech_score -= 0.4  # overbought -> slight negative
    tech_score = max(-1.0, min(1.0, tech_score))

    # news score: use aggregate_sentiment
    news_score = float(metrics.get('news_aggregate_sentiment') or 0.0)

    # model score: use analysis confidence normalized to -1..1 around 0.5
    conf = float(analysis.get('confidence') or 0.0)
    model_score = (conf - 0.5) * 2.0

    # analyst: simple heuristic from analyst_sentiment
    analyst = metrics.get('analyst_sentiment') or {}
    counts = analyst.get('counts') if isinstance(analyst, dict) else None
    analyst_score = 0.0
    if counts:
        total = sum(counts.values()) or 1
        analyst_score = (counts.get('buy', 0) - counts.get('sell', 0)) / total

    score = (
        weights['technical'] * tech_score + weights['news'] * news_score + weights['model'] * model_score + weights['analyst'] * analyst_score
    )
    suggested = (analysis.get('suggested_action') or '').lower() if isinstance(analysis, dict) else ''
    if suggested == 'buy':
        score += weights.get('buy_boost', 0.5)

    return float(score)


def run(symbols=None, output_dir='./data'):
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA']
    os.makedirs(output_dir, exist_ok=True)
    data = fetch_eod(symbols, output_dir=output_dir)
    reports = {}
    metrics_map = {}
    for s, df in data.items():
        if df is None:
            reports[s] = 'no data'
            continue
        df['SMA_14'] = add_sma(df, 14)
        df['EMA_14'] = add_ema(df, 14)
        df['RSI_14'] = add_rsi(df, 14)
        macd = add_macd(df)
        bb = add_bbands(df)
        df = pd.concat([df, macd, bb], axis=1)
        latest = df.iloc[-1]
        articles, agg_sent = aggregate_news_for_symbol(s, limit=6)
        analyst = fetch_analyst_sentiment(s)

        metrics = {
            'close': float(latest['Close']),
            'sma_14': float(latest['SMA_14']) if not pd.isna(latest['SMA_14']) else None,
            'ema_14': float(latest['EMA_14']) if not pd.isna(latest['EMA_14']) else None,
            'rsi_14': float(latest['RSI_14']) if not pd.isna(latest['RSI_14']) else None,
            'news_articles': articles,
            'news_aggregate_sentiment': agg_sent,
            'analyst_sentiment': analyst,
        }
        try:
            analysis = analyze_summary(s, metrics)
        except Exception as e:
            analysis = f'analysis failed: {e}'
        reports[s] = analysis
        metrics_map[s] = metrics
        # write individual report (markdown)
        md_path = os.path.join(output_dir, f"{s}_report.md")
        with open(md_path, 'w') as f:
            f.write(f"# {s} EOD Report\n\n")
            if isinstance(analysis, dict):
                f.write(analysis.get('summary', ''))
                f.write('\n\n')
                f.write('## Signals\n')
                f.write(str(analysis.get('signals', {})))
                f.write('\n\n')
                f.write('## Suggested Action\n')
                f.write(str(analysis.get('suggested_action')))
                f.write('\n\n')
                f.write('## Rationale\n')
                f.write(str(analysis.get('rationale')))
            else:
                f.write(str(analysis))
        # write JSON
        with open(os.path.join(output_dir, f"{s}_report.json"), 'w') as jf:
            import json as _json

            _json.dump(analysis, jf, indent=2)
    # write consolidated report
    with open(os.path.join(output_dir, 'eod_summary.md'), 'w') as f:
        for s, r in reports.items():
            f.write(f"## {s}\n\n{r}\n\n")
    # Optionally pick the best stock and invest
    invest_enabled = os.getenv('INVEST_BEST', 'false').lower() in ('1', 'true', 'yes')
    place_enabled = os.getenv('PLACE_ORDER', 'false').lower() in ('1', 'true', 'yes')
    invest_yes = os.getenv('INVEST_YES', 'false').lower() in ('1', 'true', 'yes')

    investment_result = None
    if invest_enabled:
        # score candidates using compute_candidate_score
        candidates = []
        for s, a in reports.items():
            if not isinstance(a, dict):
                continue
            metrics = metrics_map.get(s, {})
            score = compute_candidate_score(a, metrics)
            candidates.append((score, s, a.get('suggested_action'), float(a.get('confidence') or 0.0), float(metrics.get('news_aggregate_sentiment') or 0.0)))
        candidates.sort(reverse=True, key=lambda x: x[0])
        if candidates:
            best = candidates[0]
            _, best_sym, best_suggested, best_conf, best_news = best
            investment_result = {'symbol': best_sym, 'suggested_action': best_suggested, 'confidence': best_conf, 'news_sentiment': best_news}
            print(f"[INVEST] best candidate: {investment_result}")
            # only place order if PLACE_ORDER and INVEST_YES explicitly set
            # require model explicitly recommends 'buy' before placing
            if best_suggested != 'buy':
                investment_result['note'] = 'no buy recommendation from model; will not place order'
            elif place_enabled and invest_yes:
                # perform safety checks and place conservative market order
                key = os.getenv('ALPACA_KEY')
                secret = os.getenv('ALPACA_SECRET')
                if not key or not secret:
                    investment_result['error'] = 'ALPACA_KEY/SECRET missing'
                else:
                    base = 'https://paper-api.alpaca.markets' if os.getenv('ALPACA_PAPER', 'true').lower() in ('1', 'true', 'yes') else 'https://api.alpaca.markets'
                    headers = {
                        'APCA-API-KEY-ID': key,
                        'APCA-API-SECRET-KEY': secret,
                        'Content-Type': 'application/json',
                    }
                    # get account buying power
                    try:
                        r = requests.get(f"{base}/v2/account", headers=headers, timeout=10)
                        r.raise_for_status()
                        acct = r.json()
                        buying_power = float(acct.get('buying_power') or acct.get('cash') or 0.0)
                    except Exception as e:
                        investment_result['error'] = f'failed to fetch account: {e}'
                        buying_power = 0.0

                    if buying_power and buying_power > 0:
                        # size: INVEST_SIZE_PCT of buying power (default 1%)
                        pct = float(os.getenv('INVEST_SIZE_PCT', '0.01'))
                        target_usd = buying_power * pct
                        # get latest price from fetched data
                        df = data.get(best_sym)
                        if df is None or df.empty:
                            investment_result['error'] = 'no price data for symbol'
                        else:
                            latest_price = float(df.iloc[-1]['Close'])
                            qty = int(math.floor(target_usd / latest_price))
                            if qty < 1:
                                investment_result['error'] = f'calculated qty < 1 (target_usd={target_usd}, price={latest_price})'
                            else:
                                order = {'symbol': best_sym, 'qty': qty, 'side': 'buy', 'type': 'market', 'time_in_force': 'day'}
                                try:
                                    orr = requests.post(f"{base}/v2/orders", headers=headers, json=order, timeout=10)
                                    orr.raise_for_status()
                                    resp = orr.json()
                                    investment_result['order'] = resp
                                except Exception as e:
                                    investment_result['error'] = f'order failed: {e}'
                    else:
                        investment_result['error'] = 'no buying power'
        else:
            print('[INVEST] no candidates found')

    # include OpenAI call counters if available
    try:
        import eod_fetcher as _ef
        investment_result = investment_result or {}
        investment_result['openai_single_calls'] = getattr(_ef, '_OPENAI_SINGLE_CALLS', 0)
        investment_result['openai_batch_calls'] = getattr(_ef, '_OPENAI_BATCH_CALLS', 0)
    except Exception:
        pass

    # interactive confirmation if INVEST_FORCE is not set and an order would be placed
    invest_force = os.getenv('INVEST_FORCE', 'false').lower() in ('1', 'true', 'yes')
    if investment_result and investment_result.get('order') is None and invest_enabled and place_enabled and not invest_force:
        # if we would place an order based on buy recommendation, ask user (only in interactive runs)
        if investment_result.get('suggested_action') == 'buy' and investment_result.get('note') is None and investment_result.get('error') is None:
            try:
                ans = input("Confirm placing the buy order for {}? Type 'yes' to confirm: ".format(investment_result.get('symbol'))).strip().lower()
            except EOFError:
                ans = ''
            if ans != 'yes':
                investment_result['note'] = 'user declined confirmation'

    # write investment result if any
    if investment_result is not None:
        with open(os.path.join(output_dir, 'investment_action.json'), 'w') as jf:
            json.dump(investment_result, jf, indent=2)

    return reports


if __name__ == '__main__':
    print(run())
