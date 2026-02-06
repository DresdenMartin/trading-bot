import os
import json
from unittest.mock import patch, MagicMock
import requests
import scheduled_trader


def make_account(buying_power=100000, cash=100000):
    return {"buying_power": str(buying_power), "cash": str(cash)}


def test_normalize_order_resp_with_requests_response():
    resp = requests.Response()
    resp.status_code = 422
    resp._content = b'{"error":"unprocessable"}'
    normalized = scheduled_trader._normalize_order_resp(resp)
    assert normalized['status_code'] == 422
    assert normalized['body']['error'] == 'unprocessable'


@patch('analyze_with_chatgpt.analyze_portfolio_short_term')
@patch('scheduled_trader.requests.post')
@patch('scheduled_trader.requests.get')
@patch('scheduled_trader.fetch_eod')
@patch('scheduled_trader.aggregate_news_for_symbol')
def test_reallocate_full_flow(mock_agg, mock_fetch_eod, mock_get, mock_post, mock_score):
    # prepare environment
    os.environ['ALPACA_KEY'] = 'fake'
    os.environ['ALPACA_SECRET'] = 'fake'
    os.environ['ALPACA_PAPER'] = '1'
    os.environ['PLACE_ORDER'] = '0'  # start with dry-run
    os.environ['REALLOCATE_FULL'] = '1'
    os.environ['INVEST_TOTAL_PCT'] = '0.02'  # 2% for test

    # fake data for symbols: prices
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    # create simple DataFrames with Close column for latest price
    import pandas as pd
    data = {}
    for s, p in [('AAPL', 150.0), ('MSFT', 300.0), ('GOOG', 2500.0), ('AMZN', 3500.0)]:
        df = pd.DataFrame({'Close': [p]})
        data[s] = df
    mock_fetch_eod.return_value = data

    # positions: hold AAPL 10 shares and an outsider symbol XYZ to be sold
    positions = [{ 'symbol': 'AAPL', 'qty': '10' }, { 'symbol': 'XYZ', 'qty': '5' }]

    # mock account response
    account_resp = make_account(buying_power=50000, cash=10000)

    # configure requests.get to return account or positions depending on path
    def fake_get(url, headers=None, timeout=10):
        m = MagicMock()
        if url.endswith('/v2/account'):
            m.raise_for_status = lambda: None
            m.json.return_value = account_resp
        elif url.endswith('/v2/positions'):
            m.raise_for_status = lambda: None
            m.json.return_value = positions
        else:
            m.raise_for_status = lambda: None
            m.json.return_value = {}
        return m

    mock_get.side_effect = fake_get

    # aggregate news mocked
    mock_agg.return_value = ([], 0.0)

    # make analyze_portfolio_short_term return predictable ranking with symbols matching our data
    mock_score.return_value = [
        {'symbol': 'AAPL', 'score': 90},
        {'symbol': 'MSFT', 'score': 80},
        {'symbol': 'GOOG', 'score': 70},
        {'symbol': 'AMZN', 'score': 60},
    ]

    # run the Mag-7 analysis workflow
    assert scheduled_trader.analyze_top10_and_invest is scheduled_trader.analyze_mag7_and_invest
    out = scheduled_trader.analyze_mag7_and_invest()

    assert 'reallocation_plan' in out
    plan = out['reallocation_plan']
    # sells should include XYZ
    sells = plan['sells']
    assert any(s['symbol'] == 'XYZ' for s in sells)

    # buys should be non-empty (since REALLOCATE_FULL true) and target symbols are top allocations
    buys = plan['buys']
    assert isinstance(buys, list)
    assert len(buys) == 3
    guard_summary = (out.get('summary') or {}).get('guard')
    assert guard_summary is not None
    assert 'failed' in guard_summary

    # now test placing orders: enable PLACE_ORDER and patch requests.post to capture payload
    os.environ['PLACE_ORDER'] = '1'
    captured_posts = []

    def fake_post(url, headers=None, json=None, timeout=10):
        m = MagicMock()
        # simulate success
        m.raise_for_status = lambda: None
        m.json.return_value = {'id': 'order123', 'client_order_id': json.get('client_order_id')}
        captured_posts.append({'url': url, 'json': json})
        return m

    mock_post.side_effect = fake_post

    out2 = scheduled_trader.analyze_mag7_and_invest()
    # ensure post was called for buys
    assert any('/v2/orders' in p['url'] for p in captured_posts)
    # ensure at least one buy payload contains client_order_id
    assert any(p['json'] and 'client_order_id' in p['json'] for p in captured_posts)
    # ensure order summaries and post-trade validation exist
    summary = out2.get('summary') or {}
    assert 'orders' in summary
    assert summary['orders']
    assert 'post_trade' in summary
    assert summary['post_trade']['positions']


@patch('eod_fetcher.fetch_premarket_info')
@patch('scheduled_trader.fetch_premarket_info')
@patch('scheduled_trader.analyze_and_choose')
@patch('scheduled_trader.requests.post')
@patch('scheduled_trader.requests.get')
@patch('scheduled_trader.fetch_alpaca_account_and_positions')
def test_invest_flow_extended_hours_toggle(mock_fetch_acct, mock_get, mock_post, mock_analyze, mock_sched_premarket, mock_eod_premarket):
    os.environ['ALPACA_KEY'] = 'fake'
    os.environ['ALPACA_SECRET'] = 'fake'
    os.environ['ALPACA_PAPER'] = '1'
    os.environ['PLACE_ORDER'] = '1'
    os.environ['EXTENDED_HOURS'] = '1'
    os.environ['EXTENDED_HOURS_ALWAYS'] = '1'
    os.environ['INVEST_FORCE'] = '1'

    mock_fetch_acct.return_value = ({'buying_power': '100000', 'cash': '100000'}, [])
    def fake_get(url, headers=None, timeout=10):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        if url.endswith('/v2/account'):
            resp.json.return_value = {'buying_power': '100000', 'cash': '100000'}
        elif url.endswith('/v2/positions'):
            resp.json.return_value = []
        else:
            resp.json.return_value = {}
        return resp

    mock_get.side_effect = fake_get
    mock_sched_premarket.return_value = {'dollar_volume': 5_000_000, 'spread': 0.001}
    mock_eod_premarket.return_value = {'dollar_volume': 5_000_000, 'spread': 0.001}
    mock_analyze.return_value = [
        {
            'symbol': 'AAPL',
            'suggested': 'buy',
            'score': 80,
            'confidence': 0.7,
            'latest_price': 150.0,
            'analysis': {},
        }
    ]

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=10):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json.return_value = {
            'id': 'order123',
            'client_order_id': json.get('client_order_id'),
            'submitted_at': '2025-01-01T00:00:00Z',
            'filled_at': '2025-01-01T00:00:05Z',
            'filled_avg_price': '151.0',
            'status': 'filled',
        }
        captured['payload'] = json
        return resp

    mock_post.side_effect = fake_post

    out = scheduled_trader.invest_flow()

    payload = captured.get('payload')
    assert payload is not None
    assert payload.get('type') == 'limit'
    assert payload.get('extended_hours') is True
    assert payload.get('limit_price') is not None
    assert payload.get('limit_price') > 150

    summary = out.get('summary') or {}
    orders_summary = summary.get('orders') or []
    assert orders_summary
    first = orders_summary[0]
    assert first['fill_time_seconds'] == 5.0
    assert first['slippage'] == 1.0
